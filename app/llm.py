"""LLM service using Google Gemini (google-genai SDK), streaming tokens."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

# Top-level imports satisfy Pylance/Pyright static analysis.
# google-genai is a lightweight import (no model weights), so it is safe here.
from google import genai
from google.genai import types

from .config import Settings

logger = logging.getLogger("voice_assistant")


class LLMService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: genai.Client | None = None

    def load(self) -> genai.Client:
        if self._client is not None:
            return self._client

        api_key = self._settings.gemini_api_key.strip()
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY / GEMINI_API_KEY is not set. "
                "Add it to your .env file "
                "(create one at https://aistudio.google.com/apikey)."
            )
        logger.info(
            "Initializing Gemini client (model=%s)…", self._settings.gemini_model
        )
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=self._settings.gemini_http_timeout_ms,
                # The SDK's DEFAULT retry (5 attempts, exponential backoff up to
                # 60 s) silently turns one transient 429/5xx into a ~15 s stall
                # before the first token. Keep retries fast; fallback/racing
                # handles failures instead.
                retry_options=types.HttpRetryOptions(
                    attempts=self._settings.gemini_retry_attempts,
                    initial_delay=self._settings.gemini_retry_initial_delay,
                    max_delay=self._settings.gemini_retry_max_delay,
                ),
            ),
        )
        return self._client

    @staticmethod
    def _translate(messages: list[dict]) -> list:
        """Translate simple {role, content} dicts into Gemini Content objects."""
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            text = msg.get("content", "")
            # Gemini uses "model" for the assistant role (not "assistant").
            if role == "assistant":
                role = "model"
            contents.append(types.Content(role=role, parts=[types.Part(text=text)]))
        return contents

    def _candidate_models(self) -> list:
        models = [self._settings.gemini_model]
        for fb in getattr(self._settings, "gemini_fallback_models", []):
            if fb not in models:
                models.append(fb)
        return models

    async def _stream_model(
        self,
        client: genai.Client,
        model_name: str,
        config: types.GenerateContentConfig,
        history: list,
        final_text: str,
    ) -> AsyncIterator[str]:
        """Stream tokens from a single model via the chat API."""
        chat = client.aio.chats.create(
            model=model_name,
            config=config,
            history=history or None,
        )
        async for chunk in await chat.send_message_stream(final_text):
            text = getattr(chunk, "text", None)
            if text:
                yield text

    async def _stream_raced(
        self,
        client: genai.Client,
        config: types.GenerateContentConfig,
        history: list,
        final_text: str,
        candidates: list,
    ) -> AsyncIterator[str]:
        """Race the primary and first fallback, streaming whoever talks first.

        Measured flash-lite time-to-first-token swings between ~0.9 s and 15 s
        depending on Google's pool/load. Firing two models on independent
        capacity and taking the first token collapses that tail: the loser is
        cancelled the moment the winner speaks. Both requests are tiny and
        flash-lite is the cost-effective tier, so the 2x call volume is noise.
        """
        racers = candidates[:2]

        async def first_token(name: str):
            gen = self._stream_model(client, name, config, history, final_text)
            try:
                token = await anext(gen)
            except StopAsyncIteration:
                token = None
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                return name, gen, None, exc
            return name, gen, token, None

        tasks = [asyncio.create_task(first_token(n)) for n in racers]
        loser_tasks: set[asyncio.Task[Any]] = set()
        try:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for t in pending:
                t.cancel()
            loser_tasks = pending

            winner_err = None
            yielded = False
            errored = False
            for t in done:
                name, gen, token, err = t.result()
                if err is not None:
                    winner_err = err
                    errored = True
                    await gen.aclose()
                    continue
                if token is not None:
                    winner_err = None
                    errored = False
                    yielded = True
                    yield token
                    async for text in gen:
                        yield text
                    break
                await gen.aclose()

            await asyncio.gather(*pending, return_exceptions=True)
        finally:
            for t in list(loser_tasks):
                if not t.done():
                    t.cancel()
            still = [t for t in list(loser_tasks) if not t.done()]
            if still:
                await asyncio.gather(*still, return_exceptions=True)

        if yielded:
            return

        # Neither racer produced a token. Give any remaining candidate capacity
        # one sequential shot before failing.
        for name in candidates[2:]:
            try:
                async for text in self._stream_model(
                    client, name, config, history, final_text
                ):
                    yield text
                return
            except Exception as exc:  # noqa: BLE001
                winner_err = exc
                logger.warning(
                    "LLM model %s failed before tokens (%s); trying fallback...",
                    name,
                    exc,
                )

        if not errored and winner_err is None:
            return  # empty reply (unchanged output), e.g. prompt filtered

        if winner_err is not None:
            raise winner_err

    async def stream_reply(self, messages: list[dict]) -> AsyncIterator[str]:
        """Stream LLM reply tokens one by one with resilient fallback.

        Uses the chat API (`chats.create` + `send_message_stream`), which the
        genai SDK recommends over `models.generate_content_stream` and which we
        measured ~16x faster to first token (900 ms vs 15+ s) on the flash-lite
        tier.

        The whole generation is bounded by ``llm_timeout_seconds`` so a hung or
        silently-retrying upstream surfaces as a fast, clear error instead of a
        silent stall. TimeoutError while suspended is raised at the next chunk
        and caught by the caller's exceptions handling.
        """
        client = self.load()

        config = types.GenerateContentConfig(
            system_instruction=self._settings.gemini_system_prompt,
            temperature=0.6,
            max_output_tokens=self._settings.llm_max_output_tokens,
        )
        contents = self._translate(messages)
        history = contents[:-1]
        final_text = contents[-1].parts[0].text if contents else ""

        if not final_text:
            return

        candidates = self._candidate_models()

        # Each model gets its OWN time budget (split of llm_timeout_seconds).
        # Wrapping the whole loop in one timeout starved the fallbacks: a slow
        # primary consumed the entire budget, so the fast fallback never got to
        # answer. Now a throttled primary yields to the next candidate. The
        # primary gets a bigger slice (~2x a fallback) so it normally wins.
        total_deadline = (
            asyncio.get_running_loop().time() + self._settings.llm_timeout_seconds
        )
        num_candidates = max(1, len(candidates))
        weights = [2.0] + [1.0] * (num_candidates - 1)
        base_unit = self._settings.llm_timeout_seconds / sum(weights)

        last_err: Exception | None = None
        if self._settings.gemini_race_fallback and len(candidates) >= 2:
            async with asyncio.timeout(self._settings.llm_timeout_seconds):
                async for text in self._stream_raced(
                    client, config, history, final_text, candidates
                ):
                    yield text
            return

        for i, model_name in enumerate(candidates):
            remaining = total_deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            # Last candidate may use whatever remains of the overall budget.
            budget = (
                remaining
                if i == num_candidates - 1
                else min(base_unit * weights[i], remaining)
            )
            tokens_yielded = False
            try:
                async with asyncio.timeout(budget):
                    async for text in self._stream_model(
                        client, model_name, config, history, final_text
                    ):
                        tokens_yielded = True
                        yield text
                return
            except TimeoutError:
                if tokens_yielded:
                    # Tokens already reached the user; switching models now would
                    # splice two replies together. Surface the failure instead.
                    logger.error(
                        "LLM stream timed out mid-generation on %s.", model_name
                    )
                    raise
                last_err = TimeoutError(f"{model_name} timed out after {budget:.1f}s")
                logger.warning(
                    "LLM model %s timed out after %.1fs; trying fallback...",
                    model_name,
                    budget,
                )
                continue
            except Exception as exc:
                last_err = exc
                if tokens_yielded:
                    logger.error(
                        "LLM stream broke mid-generation on %s: %s", model_name, exc
                    )
                    raise
                logger.warning(
                    "LLM model %s failed before tokens (%s); trying fallback...",
                    model_name,
                    exc,
                )
                continue

        if last_err:
            raise last_err
