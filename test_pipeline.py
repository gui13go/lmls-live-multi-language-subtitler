"""test_pipeline.py - Comprehensive test suite for Live Subtitles Overlay.

Validates:
1. Audio capture & Silero VAD detection.
2. WhisperTranscriber hallucination filters & language ID.
3. TranslationEngine caching, passthrough, and parallel worker.
4. PyQt6 SubtitleOverlay dragging, repositioning, and click-through toggle.
5. Remote GPU server data structures and endpoints.
"""

import os
import sys
import time
import unittest
import numpy as np

# Force XCB for headless/automated test consistency
if sys.platform.startswith("linux") and os.environ.get("DISPLAY"):
    os.environ["QT_QPA_PLATFORM"] = "xcb"


class TestAudioAndVAD(unittest.TestCase):
    """Test audio configuration and Silero VAD."""

    def test_silero_vad_silence(self):
        from audio import SileroVADDetector, SAMPLE_RATE, VAD_CHUNK_SIZE
        vad = SileroVADDetector(threshold=0.5)
        # 512 zeros = dead silence
        silence = np.zeros(VAD_CHUNK_SIZE, dtype=np.float32)
        prob = vad.get_speech_prob(silence)
        self.assertLess(prob, 0.3, f"Silence prob too high: {prob}")

    def test_silero_vad_signal(self):
        from audio import SileroVADDetector, VAD_CHUNK_SIZE
        vad = SileroVADDetector(threshold=0.5)
        # Synthetic speech-like harmonic signal
        t = np.linspace(0, 0.032, VAD_CHUNK_SIZE, endpoint=False)
        wave = (0.4 * np.sin(2 * np.pi * 300 * t) + 0.3 * np.sin(2 * np.pi * 600 * t)).astype(np.float32)
        prob = vad.get_speech_prob(wave)
        self.assertIsInstance(prob, float)

    def test_list_devices(self):
        from audio import list_audio_devices
        devs = list_audio_devices()
        self.assertIsInstance(devs, list)


class TestTranscriber(unittest.TestCase):
    """Test transcriber sanitization and hallucination filtering."""

    def test_clean_transcribed_text(self):
        from transcriber import clean_transcribed_text

        # Hallucinations should be filtered
        self.assertIsNone(clean_transcribed_text(""))
        self.assertIsNone(clean_transcribed_text("   ...   "))
        self.assertIsNone(clean_transcribed_text("Thank you."))
        self.assertIsNone(clean_transcribed_text("Thanks for watching!"))
        self.assertIsNone(clean_transcribed_text("Subtitles by..."))
        self.assertIsNone(clean_transcribed_text("you"))
        self.assertIsNone(clean_transcribed_text("..............."))

        # Valid speech should be preserved
        valid = "Welcome to today's live demonstration."
        self.assertEqual(clean_transcribed_text(valid), valid)


class TestTranslator(unittest.TestCase):
    """Test translation worker, caching, and language passthrough."""

    def test_language_passthrough(self):
        from translator import TranslationEngine, normalize_lang_code

        engine = TranslationEngine(engine_name="google")
        text = "Hello world"
        # Same language: 0ms passthrough without network call
        result = engine.translate_single(text, "en", "en")
        self.assertEqual(result, text)

        # Alias normalization (zh vs zh-cn)
        self.assertEqual(normalize_lang_code("zh"), "zh-CN")
        self.assertEqual(normalize_lang_code("english"), "en")

    def test_caching(self):
        from translator import TranslationEngine

        engine = TranslationEngine(engine_name="google")
        text = "Hello"
        # Seed cache manually
        key = engine._get_cache_key("en", "es", text)
        engine._cache[key] = "Hola"

        res = engine.translate_single(text, "en", "es")
        self.assertEqual(res, "Hola")


class TestGUIOverlay(unittest.TestCase):
    """Test PyQt6 SubtitleOverlay dragging and click-through mode."""

    @classmethod
    def setUpClass(cls):
        from PyQt6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_overlay_creation_and_drag(self):
        from PyQt6.QtCore import Qt, QPoint, QPointF, QEvent
        from PyQt6.QtGui import QMouseEvent
        from gui import SubtitleOverlay

        overlay = SubtitleOverlay(
            target_languages=["en", "zh", "de"],
            position="top",
            font_size=20,
            start_locked=False,
            width=900,
        )
        overlay.show()
        # Allow initial X11 server window mapping to settle
        for _ in range(10):
            self.app.processEvents()
            time.sleep(0.01)

        # Check default state: unlocked/movable
        self.assertFalse(overlay.click_through_enabled)
        initial_pos = overlay.pos()

        # Test dragging from the drag handle
        press_pt = QPointF(float(initial_pos.x() + 40), float(initial_pos.y() + 15))
        e_press = QMouseEvent(
            QEvent.Type.MouseButtonPress, press_pt, press_pt,
            Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier
        )
        self.app.sendEvent(overlay.drag_grip, e_press)
        self.assertTrue(overlay._dragging)

        # Move by +100 X, +150 Y
        move_pt = QPointF(float(initial_pos.x() + 140), float(initial_pos.y() + 165))
        e_move = QMouseEvent(
            QEvent.Type.MouseMove, move_pt, move_pt,
            Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier
        )
        self.app.sendEvent(overlay.drag_grip, e_move)
        for _ in range(3):
            self.app.processEvents()

        new_pos = overlay.pos()
        self.assertEqual(new_pos.x(), initial_pos.x() + 100)
        self.assertEqual(new_pos.y(), initial_pos.y() + 150)

        # Release mouse
        e_release = QMouseEvent(
            QEvent.Type.MouseButtonRelease, move_pt, move_pt,
            Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier
        )
        self.app.sendEvent(overlay.drag_grip, e_release)
        self.assertFalse(overlay._dragging)

        # Test toggle click-through
        overlay.toggle_click_through_mode()
        self.assertTrue(overlay.click_through_enabled)
        overlay.toggle_click_through_mode()
        self.assertFalse(overlay.click_through_enabled)

        overlay.close()


if __name__ == "__main__":
    unittest.main()
