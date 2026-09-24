"""translator.py - Decoupled, parallel multi-target translation worker.

Routes spoken language dynamically: passes through matching target languages directly
without translation, and translates non-matching targets in parallel via a fast,
non-blocking translation backend (e.g., Google, DeepL, MyMemory via deep-translator).
"""

from __future__ import annotations

import collections
import concurrent.futures
import hashlib
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set

logger = logging.getLogger("live_subtitles.translator")


@dataclass
class TranslationBundle:
    """Bundle containing original transcription and all target translations."""

    source_lang: str
    original_text: str
    translations: Dict[str, str]  # {lang_code: translated_text}
    timestamp: float
    latency_ms: float


# ISO language code normalization mapping
LANG_ALIASES: Dict[str, str] = {
    "zh-cn": "zh-CN",
    "zh-tw": "zh-TW",
    "zh": "zh-CN",
    "chinese": "zh-CN",
    "english": "en",
    "german": "de",
    "spanish": "es",
    "french": "fr",
    "japanese": "ja",
    "korean": "ko",
    "italian": "it",
    "portuguese": "pt",
    "russian": "ru",
}


def normalize_lang_code(code: str) -> str:
    """Normalize language codes for uniform comparison and backend compatibility."""
    lowered = code.strip().lower()
    return LANG_ALIASES.get(lowered, lowered)


MYMEMORY_MAP: Dict[str, str] = {
    "en": "en-US",
    "de": "de-DE",
    "es": "es-ES",
    "fr": "fr-FR",
    "it": "it-IT",
    "ja": "ja-JP",
    "ko": "ko-KR",
    "pt": "pt-PT",
    "ru": "ru-RU",
    "zh": "zh-CN",
    "zh-cn": "zh-CN",
    "zh-tw": "zh-TW",
    "nl": "nl-NL",
    "pl": "pl-PL",
    "sv": "sv-SE",
    "tr": "tr-TR",
    "ar": "ar-SA",
    "hi": "hi-IN",
}


def to_mymemory_code(code: str) -> str:
    """Map ISO language code to MyMemory RFC-5646 format."""
    clean = code.lower().strip()
    return MYMEMORY_MAP.get(clean, MYMEMORY_MAP.get(clean.split("-")[0], clean))


class TranslationEngine:
    """Handles single language pair translations with in-memory caching and fallback."""

    def __init__(self, engine_name: str = "google", api_key: Optional[str] = None):
        self.engine_name = engine_name.lower()
        self.api_key = api_key
        # LRU cache: (source_lang, target_lang, md5) -> translated_text
        self._cache: collections.OrderedDict[Tuple[str, str, str], str] = (
            collections.OrderedDict()
        )
        self._cache_lock = threading.Lock()
        self._max_cache_size = 1000
        # If primary engine rate limits or fails, auto-switch to avoid redundant latency
        self._google_disabled = (self.engine_name == "mymemory")

    def _get_cache_key(self, src: str, tgt: str, text: str) -> Tuple[str, str, str]:
        h = hashlib.md5(text.encode("utf-8")).hexdigest()
        return (src, tgt, h)

    def translate_single(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate a single piece of text from source to target.

        Uses caching and falls back across engines if needed.
        """
        if not text or not text.strip():
            return ""

        src_norm = normalize_lang_code(source_lang)
        tgt_norm = normalize_lang_code(target_lang)

        # Same language: direct passthrough
        if src_norm == tgt_norm or src_norm.split("-")[0] == tgt_norm.split("-")[0]:
            return text

        cache_key = self._get_cache_key(src_norm, tgt_norm, text)
        with self._cache_lock:
            if cache_key in self._cache:
                self._cache.move_to_end(cache_key)
                return self._cache[cache_key]

        translated = self._execute_translation(text, src_norm, tgt_norm)

        with self._cache_lock:
            self._cache[cache_key] = translated
            if len(self._cache) > self._max_cache_size:
                self._cache.popitem(last=False)

        return translated

    def _execute_translation(self, text: str, src: str, tgt: str) -> str:
        """Execute translation via deep-translator backends."""
        # 1. DeepL if configured
        if self.engine_name == "deepl" and self.api_key:
            try:
                from deep_translator import DeeplTranslator

                tr = DeeplTranslator(api_key=self.api_key, source=src, target=tgt)
                return tr.translate(text)
            except Exception as e_deepl:
                logger.warning(f"DeepL translation error: {e_deepl}")

        # 2. Google Translator (if not disabled due to 429)
        if not self._google_disabled:
            try:
                from deep_translator import GoogleTranslator

                tr = GoogleTranslator(source=src, target=tgt)
                res = tr.translate(text)
                if res:
                    return res
            except Exception as exc:
                logger.debug(
                    f"Google Translator failed ({exc}). Failing over to MyMemory."
                )
                self._google_disabled = True

        # 3. MyMemoryTranslator fallback
        try:
            from deep_translator import MyMemoryTranslator

            mm_src = to_mymemory_code(src)
            mm_tgt = to_mymemory_code(tgt)
            tr = MyMemoryTranslator(source=mm_src, target=mm_tgt)
            res = tr.translate(text)
            if res:
                return res
        except Exception as fb_exc:
            logger.debug(f"MyMemory fallback failed for {src}->{tgt}: {fb_exc}")

        # 4. Graceful fallback: return original text if translation unavailable
        return f"[{tgt.upper()}] {text}"


class TranslationWorker(threading.Thread):
    """Asynchronous worker that receives transcription events, coordinates parallel

    translations across all configured target languages, and dispatches finished bundles.
    """

    def __init__(
        self,
        target_languages: List[str],
        result_callback: Callable[[TranslationBundle], None],
        engine_name: str = "google",
        api_key: Optional[str] = None,
        max_workers: Optional[int] = None,
    ):
        super().__init__(daemon=True, name="TranslationWorkerThread")
        self.target_languages = [normalize_lang_code(t) for t in target_languages]
        self.result_callback = result_callback
        self.engine = TranslationEngine(engine_name=engine_name, api_key=api_key)

        pool_size = max_workers or max(2, len(self.target_languages))
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=pool_size, thread_name_prefix="TransSubWorker"
        )
        self._input_queue: queue.Queue[Any] = queue.Queue(maxsize=100)
        self._stop_event = threading.Event()

    def submit_transcription(self, result: Any) -> None:
        """Push a TranscriptionResult for translation."""
        try:
            self._input_queue.put_nowait(result)
        except queue.Full:
            logger.warning("Translation input queue full. Dropped item.")

    def stop(self) -> None:
        self._stop_event.set()
        self.executor.shutdown(wait=False, cancel_futures=True)

    def run(self) -> None:
        logger.info(
            f"Translation worker active with target languages: {self.target_languages}"
        )
        while not self._stop_event.is_set():
            try:
                item = self._input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                self._process_item(item)
            except Exception as exc:
                logger.error(f"Error processing translation bundle: {exc}", exc_info=True)
            finally:
                self._input_queue.task_done()

    def _process_item(self, item: Any) -> None:
        """Dispatch parallel translation tasks for each target language."""
        start_time = time.time()
        spoken_lang = normalize_lang_code(item.language)
        original_text = item.text

        translations: Dict[str, str] = {}
        futures_map: Dict[concurrent.futures.Future, str] = {}

        for tgt in self.target_languages:
            # Check for direct language match
            if tgt == spoken_lang or tgt.split("-")[0] == spoken_lang.split("-")[0]:
                translations[tgt] = original_text
            else:
                # Submit translation job to thread pool
                future = self.executor.submit(
                    self.engine.translate_single, original_text, spoken_lang, tgt
                )
                futures_map[future] = tgt

        # Gather results concurrently
        if futures_map:
            done, not_done = concurrent.futures.wait(
                futures_map.keys(), timeout=3.0
            )
            for fut in done:
                tgt = futures_map[fut]
                try:
                    translations[tgt] = fut.result()
                except Exception as exc:
                    logger.warning(f"Translation failed for target '{tgt}': {exc}")
                    translations[tgt] = f"[{tgt.upper()}] {original_text}"

            for fut in not_done:
                tgt = futures_map[fut]
                translations[tgt] = f"[{tgt.upper()}] {original_text}"

        elapsed_ms = (time.time() - start_time) * 1000.0

        bundle = TranslationBundle(
            source_lang=spoken_lang,
            original_text=original_text,
            translations=translations,
            timestamp=item.timestamp,
            latency_ms=elapsed_ms,
        )

        logger.debug(
            f"Completed parallel translations in {elapsed_ms:.1f}ms for targets: {list(translations.keys())}"
        )
        self.result_callback(bundle)
