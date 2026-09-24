"""gui.py - Frameless, transparent, always-on-top cross-platform subtitle overlay.

Built with PyQt6. Features native Wayland/X11/Win32 drag-and-drop repositioning,
mouse click-through toggle, configurable typography, and global hotkeys.
"""

from __future__ import annotations

import logging
import os
import platform
import sys
from typing import Dict, List, Optional

# Force X11/XCB backend on Linux desktops when DISPLAY is available.
# This ensures frameless overlays have full window manager authority
# to position anywhere on screen, support click-through transparency,
# stay on top, and register global hotkeys.
if sys.platform.startswith("linux") and "QT_QPA_PLATFORM" not in os.environ and os.environ.get("DISPLAY"):
    os.environ["QT_QPA_PLATFORM"] = "xcb"

from PyQt6.QtCore import QEvent, QObject, QPoint, QRect, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QFont, QGuiApplication, QMouseEvent, QPainter, QRegion
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger("live_subtitles.gui")

# Distinct color palette for language badges
BADGE_COLORS: Dict[str, str] = {
    "en": "#0284c7",  # Sky blue
    "zh": "#e11d48",  # Rose
    "zh-cn": "#e11d48",
    "zh-tw": "#be123c",
    "de": "#d97706",  # Amber
    "es": "#059669",  # Emerald
    "fr": "#7c3aed",  # Violet
    "ja": "#dc2626",  # Crimson
    "ko": "#4f46e5",  # Indigo
    "it": "#0d9488",  # Teal
    "pt": "#ea580c",  # Orange
    "ru": "#0284c7",  # Blue
    "default": "#4b5563",  # Slate
}


class SubtitleRow(QWidget):
    """A single row displaying a language badge and its live subtitle text."""

    def __init__(self, lang_code: str, font_size: int = 20, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.lang_code = lang_code.lower()
        self.font_size = font_size

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(14)

        # Language badge
        self.badge = QLabel(f"[{lang_code.upper()}]", self)
        badge_bg = BADGE_COLORS.get(self.lang_code, BADGE_COLORS["default"])
        badge_font = QFont("Segoe UI", max(11, int(self.font_size * 0.65)), QFont.Weight.Bold)
        self.badge.setFont(badge_font)
        self.badge.setStyleSheet(
            f"background-color: {badge_bg}; color: #ffffff; "
            f"border-radius: 6px; padding: 3px 8px; font-weight: bold;"
        )
        self.badge.setFixedWidth(max(55, int(self.font_size * 2.8)))
        self.badge.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Subtitle text label
        self.text_label = QLabel("...", self)
        text_font = QFont("Segoe UI", self.font_size, QFont.Weight.DemiBold)
        self.text_label.setFont(text_font)
        self.text_label.setStyleSheet("color: #ffffff; background: transparent;")
        self.text_label.setWordWrap(True)

        # Subtle text drop shadow for legibility over bright backgrounds
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(8)
        shadow.setColor(QColor(0, 0, 0, 220))
        shadow.setOffset(1, 2)
        self.text_label.setGraphicsEffect(shadow)

        layout.addWidget(self.badge, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.text_label, stretch=1)

    def set_text(self, text: str) -> None:
        cleaned = text.strip() or "..."
        self.text_label.setText(cleaned)


class OverlaySignals(QObject):
    """Thread-safe Qt signal bridge for subtitle updates and hotkeys."""

    update_subtitles = pyqtSignal(dict)  # {lang_code: text}
    toggle_visibility = pyqtSignal()
    toggle_click_through = pyqtSignal()
    clear_subtitles = pyqtSignal()
    quit_app = pyqtSignal()


class DragEventFilter(QObject):
    """Application-level event filter ensuring drag-and-drop works seamlessly across
    all desktop environments even when clicking child widgets, labels, or frames.
    """

    def __init__(self, overlay: "SubtitleOverlay"):
        super().__init__(overlay)
        self.overlay = overlay

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        # Only handle dragging when click-through is NOT active
        if not self.overlay.click_through_enabled:
            # Let interactive buttons receive their normal clicks
            if watched in (self.overlay.lock_button, self.overlay.clear_button):
                return False

            if event.type() == QEvent.Type.MouseButtonPress:
                if isinstance(event, QMouseEvent) and event.button() == Qt.MouseButton.LeftButton:
                    self.overlay._dragging = True
                    self.overlay._drag_start_pos = (
                        event.globalPosition().toPoint() - self.overlay.pos()
                    )
                    self.overlay.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
                    return True

            elif event.type() == QEvent.Type.MouseMove:
                if self.overlay._dragging and isinstance(event, QMouseEvent):
                    new_pos = event.globalPosition().toPoint() - self.overlay._drag_start_pos
                    self.overlay.move(new_pos)
                    return True

            elif event.type() == QEvent.Type.MouseButtonRelease:
                if self.overlay._dragging and isinstance(event, QMouseEvent) and event.button() == Qt.MouseButton.LeftButton:
                    self.overlay._dragging = False
                    self.overlay.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))
                    return True

        return super().eventFilter(watched, event)


class SubtitleOverlay(QMainWindow):
    """Frameless, always-on-top, draggable, click-through transparent desktop overlay."""

    def __init__(
        self,
        target_languages: List[str],
        position: str = "top",
        font_size: int = 20,
        opacity: float = 0.72,
        start_locked: bool = False,
        width: Optional[int] = None,
    ):
        super().__init__()
        self.target_languages = [t.strip().lower() for t in target_languages]
        self.position_mode = position.lower()
        self.font_size = font_size
        self.bg_opacity = opacity
        self.custom_width = width
        self.click_through_enabled = start_locked
        self.signals = OverlaySignals()

        # Window drag state
        self._dragging = False
        self._drag_start_pos = QPoint()

        # Connect signals
        self.signals.update_subtitles.connect(self.on_update_subtitles)
        self.signals.toggle_visibility.connect(self.toggle_overlay)
        self.signals.toggle_click_through.connect(self.toggle_click_through_mode)
        self.signals.clear_subtitles.connect(self.clear_all_text)
        self.signals.quit_app.connect(self.close)

        self._setup_window_properties()
        self._setup_ui()
        self._position_overlay()
        self._apply_click_through(self.click_through_enabled)

        # Install global drag filter across the entire window hierarchy
        self._drag_filter = DragEventFilter(self)
        qApp = QApplication.instance()
        if qApp:
            qApp.installEventFilter(self._drag_filter)

        # Inactivity clear timer (25s)
        self.clear_timer = QTimer(self)
        self.clear_timer.setInterval(25000)
        self.clear_timer.timeout.connect(self._on_inactivity_timeout)

    def _setup_window_properties(self) -> None:
        """Configure Qt window flags for frameless, floating HUD."""
        flags = (
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

    def _setup_ui(self) -> None:
        """Construct the visual hierarchy of the floating HUD bar."""
        self.central_container = QFrame(self)
        self.central_container.setObjectName("CentralFrame")

        main_vbox = QVBoxLayout(self.central_container)
        main_vbox.setContentsMargins(14, 8, 14, 10)
        main_vbox.setSpacing(6)

        # Interactive Header Bar with Drag Grip & Lock Toggle
        self.header_frame = QFrame(self)
        self.header_frame.setObjectName("HeaderFrame")
        self.header_frame.setStyleSheet(
            """
            QFrame#HeaderFrame {
                background-color: rgba(255, 255, 255, 0.08);
                border-radius: 8px;
                padding: 2px 6px;
            }
            """
        )
        self.header_layout = QHBoxLayout(self.header_frame)
        self.header_layout.setContentsMargins(6, 4, 6, 4)
        self.header_layout.setSpacing(10)

        # Drag Handle Grip
        self.drag_grip = QLabel("⠿ DRAG TO MOVE", self.header_frame)
        self.drag_grip.setStyleSheet("color: #38bdf8; font-size: 11px; font-weight: bold;")
        self.header_layout.addWidget(self.drag_grip)

        # Clickable Lock / Pass-Through Toggle Button
        self.lock_button = QPushButton(self.header_frame)
        self.lock_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.lock_button.clicked.connect(self.toggle_click_through_mode)
        self.header_layout.addWidget(self.lock_button)

        self.header_layout.addStretch()

        # Clear Button
        self.clear_button = QPushButton("Clear", self.header_frame)
        self.clear_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.clear_button.setStyleSheet(
            "background-color: rgba(255, 255, 255, 0.12); color: #e2e8f0; "
            "border: none; border-radius: 4px; padding: 2px 8px; font-size: 10px;"
        )
        self.clear_button.clicked.connect(self.clear_all_text)
        self.header_layout.addWidget(self.clear_button)

        # Shortcut Hint
        self.shortcut_hint = QLabel("Ctrl+Shift+T: Lock/Unlock | Ctrl+Shift+H: Hide", self.header_frame)
        self.shortcut_hint.setStyleSheet("color: rgba(255, 255, 255, 0.5); font-size: 10px;")
        self.header_layout.addWidget(self.shortcut_hint)

        main_vbox.addWidget(self.header_frame)

        # Language subtitle rows
        self.rows: Dict[str, SubtitleRow] = {}
        for lang in self.target_languages:
            row = SubtitleRow(lang, font_size=self.font_size, parent=self)
            self.rows[lang] = row
            main_vbox.addWidget(row)

        self.setCentralWidget(self.central_container)

    def _position_overlay(self) -> None:
        """Position the overlay relative to the primary screen."""
        screen = QGuiApplication.primaryScreen()
        if not screen:
            self.resize(1000, 160)
            return

        geom = screen.geometry()
        screen_w = geom.width()
        screen_h = geom.height()

        if self.custom_width and self.custom_width > 200:
            width = min(self.custom_width, screen_w - 40)
        else:
            width = int(min(1150, max(720, screen_w * 0.65)))

        row_height = max(42, int(self.font_size * 2.1))
        height = 60 + (len(self.target_languages) * row_height)

        x = geom.x() + (screen_w - width) // 2

        if self.position_mode == "bottom":
            y = geom.y() + screen_h - height - 45
        else:
            y = geom.y() + 35

        self.setGeometry(QRect(x, y, width, height))

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Direct mouse press drag support for the overlay window."""
        if not self.click_through_enabled and event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._drag_start_pos = event.globalPosition().toPoint() - self.pos()
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Move the overlay to follow the cursor during an active drag."""
        if self._dragging and not self.click_through_enabled:
            new_pos = event.globalPosition().toPoint() - self._drag_start_pos
            self.move(new_pos)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Finish dragging and restore cursor."""
        if self._dragging and event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def _apply_click_through(self, enable: bool) -> None:
        """Configure platform-specific click-through transparency."""
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, enable)

        alpha = int(self.bg_opacity * 255)

        # Windows-specific extended window styles (WS_EX_TRANSPARENT | WS_EX_LAYERED)
        if sys.platform == "win32":
            try:
                import ctypes

                hwnd = int(self.winId())
                GWL_EXSTYLE = -20
                WS_EX_TRANSPARENT = 0x00000020
                WS_EX_LAYERED = 0x00080000

                user32 = ctypes.windll.user32
                styles = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
                if enable:
                    user32.SetWindowLongW(
                        hwnd, GWL_EXSTYLE, styles | WS_EX_TRANSPARENT | WS_EX_LAYERED
                    )
                else:
                    user32.SetWindowLongW(
                        hwnd, GWL_EXSTYLE, styles & ~WS_EX_TRANSPARENT
                    )
            except Exception as exc:
                logger.debug(f"Win32 click-through setup warning: {exc}")

        # Visual indicator and cursor updates
        if enable:
            self.central_container.setStyleSheet(
                f"""
                QFrame#CentralFrame {{
                    background-color: rgba(12, 16, 24, {alpha});
                    border: 1px solid rgba(255, 255, 255, 0.15);
                    border-radius: 14px;
                }}
                """
            )
            self.lock_button.setText("🔒 Pass-Through: ON (Locked)")
            self.lock_button.setStyleSheet(
                "background-color: #065f46; color: #34d399; border: 1px solid #10b981; "
                "border-radius: 4px; padding: 2px 8px; font-size: 11px; font-weight: bold;"
            )
            self.drag_grip.setText("⠿ (Locked)")
            self.drag_grip.setStyleSheet("color: #64748b; font-size: 11px;")
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        else:
            # Highlight border in movable mode so user sees it can be moved
            self.central_container.setStyleSheet(
                f"""
                QFrame#CentralFrame {{
                    background-color: rgba(12, 16, 24, {alpha});
                    border: 2px solid #38bdf8;
                    border-radius: 14px;
                }}
                """
            )
            self.lock_button.setText("🔓 Movable: ON (Click to Lock)")
            self.lock_button.setStyleSheet(
                "background-color: #92400e; color: #fbbf24; border: 1px solid #f59e0b; "
                "border-radius: 4px; padding: 2px 8px; font-size: 11px; font-weight: bold;"
            )
            self.drag_grip.setText("⠿ DRAG TO MOVE")
            self.drag_grip.setStyleSheet("color: #38bdf8; font-size: 11px; font-weight: bold;")
            self.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))

    def toggle_click_through_mode(self) -> None:
        """Toggle between click-through mode and movable/draggable mode."""
        self.click_through_enabled = not self.click_through_enabled
        self._apply_click_through(self.click_through_enabled)
        logger.info(f"Click-through mode set to: {self.click_through_enabled}")

    def toggle_overlay(self) -> None:
        """Toggle overlay visibility."""
        if self.isVisible():
            self.hide()
            logger.info("Overlay hidden.")
        else:
            self.show()
            self.raise_()
            logger.info("Overlay shown.")

    def clear_all_text(self) -> None:
        for row in self.rows.values():
            row.set_text("...")

    def _on_inactivity_timeout(self) -> None:
        self.clear_all_text()

    def on_update_subtitles(self, translations: Dict[str, str]) -> None:
        self.clear_timer.start()
        for lang, text in translations.items():
            lang_key = lang.lower()
            if lang_key in self.rows:
                self.rows[lang_key].set_text(text)
            else:
                prefix = lang_key.split("-")[0]
                for k, row in self.rows.items():
                    if k.split("-")[0] == prefix:
                        row.set_text(text)
                        break


def setup_global_hotkeys(signals: OverlaySignals) -> Optional[Any]:
    """Register system-wide hotkeys using pynput in a background daemon thread."""
    try:
        from pynput import keyboard

        hotkeys = {
            "<ctrl>+<shift>+h": signals.toggle_visibility.emit,
            "<ctrl>+<shift>+t": signals.toggle_click_through.emit,
            "<ctrl>+<shift>+c": signals.clear_subtitles.emit,
            "<ctrl>+<shift>+q": signals.quit_app.emit,
        }

        listener = keyboard.GlobalHotKeys(hotkeys)
        listener.daemon = True
        listener.start()
        logger.info("Global hotkeys registered successfully (Ctrl+Shift+H/T/C/Q).")
        return listener
    except Exception as exc:
        logger.warning(
            f"Could not initialize system-wide hotkeys via pynput ({exc}). "
            "In-app shortcuts will be used instead."
        )
        return None
