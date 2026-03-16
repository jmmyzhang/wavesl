"""
Interactive device selection for WaveSL.
Presents arrow-key menus in the terminal for camera and audio output selection.
A live camera preview is the authoritative way to identify the correct device.
"""

import curses
import sys
from typing import Optional

import cv2

# Sentinel returned by menus when the user presses q to quit
_QUIT = -1


# ──────────────────────────────────────────────
# Device enumeration
# ──────────────────────────────────────────────

def enumerate_cameras() -> list[tuple[int, int, int]]:
    """
    Probe OpenCV indices 0–9. Returns [(index, width, height), ...].
    Names are intentionally omitted: on macOS the AVFoundation index order
    differs from system_profiler order and cannot be reliably correlated
    without ffmpeg. Use the live preview to identify the correct camera.
    """
    cameras: list[tuple[int, int, int]] = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            cameras.append((i, w, h))
    return cameras


def enumerate_audio_outputs() -> list[tuple[int, str]]:
    """Return [(device_id, name), ...] for all audio output-capable devices."""
    try:
        import sounddevice as sd
        return [
            (i, d['name'])
            for i, d in enumerate(sd.query_devices())
            if d['max_output_channels'] > 0
        ]
    except Exception:
        return []


# ──────────────────────────────────────────────
# Camera preview confirmation
# ──────────────────────────────────────────────

def confirm_camera(index: int) -> bool:
    """
    Show a live preview window for the given camera index.
    Returns True on Enter (confirmed), False on Esc (go back).
    """
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f'  Could not open camera {index}.')
        return False

    confirmed = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        label = f'Camera {index}'
        msg = 'ENTER = use this camera    ESC = go back'
        for color, thickness in [((0, 0, 0), 4), ((255, 255, 255), 2)]:
            cv2.putText(frame, label, (20, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, thickness, cv2.LINE_AA)
        for color, thickness in [((0, 0, 0), 3), ((0, 220, 255), 2)]:
            cv2.putText(frame, msg, (20, 92),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, thickness, cv2.LINE_AA)

        cv2.imshow('WaveSL - Camera Preview', frame)
        key = cv2.waitKey(30) & 0xFF
        if key in (10, 13):
            confirmed = True
            break
        elif key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    return confirmed


# ──────────────────────────────────────────────
# Curses menu  (returns _QUIT on q)
# ──────────────────────────────────────────────

def _curses_menu(stdscr, title: str, options: list[str]) -> int:
    """
    Arrow-key curses menu.
    Returns the selected index, or _QUIT if the user presses q.
    """
    curses.curs_set(0)
    current = 0

    while True:
        stdscr.clear()
        h, w = stdscr.getmaxyx()

        stdscr.addstr(0, 0, title, curses.A_BOLD)
        stdscr.addstr(1, 0, '-' * min(w - 1, 60))

        for i, opt in enumerate(options):
            y = i + 3
            if y >= h - 2:
                break
            if i == current:
                stdscr.addstr(y, 0, '>', curses.A_BOLD)
                stdscr.addstr(y, 2, opt[:w - 3], curses.A_REVERSE)
            else:
                stdscr.addstr(y, 0, f'  {opt}'[:w - 1])

        stdscr.addstr(h - 1, 0, 'UP/DOWN: navigate   ENTER: select   q: quit'[:w - 1])
        stdscr.refresh()

        key = stdscr.getch()
        if key == curses.KEY_UP and current > 0:
            current -= 1
        elif key == curses.KEY_DOWN and current < len(options) - 1:
            current += 1
        elif key in (curses.KEY_ENTER, 10, 13):
            return current
        elif key in (ord('q'), ord('Q')):
            return _QUIT


def _fallback_menu(title: str, options: list[str]) -> int:
    """Numbered fallback if curses is unavailable (e.g. IDE terminal)."""
    print(f'\n{title}')
    for i, opt in enumerate(options):
        print(f'  [{i}] {opt}')
    print('  [q] Quit')
    while True:
        raw = input('Enter number (or q): ').strip().lower()
        if raw == 'q':
            return _QUIT
        try:
            val = int(raw)
            if 0 <= val < len(options):
                return val
        except ValueError:
            pass
        print(f'  Enter a number between 0 and {len(options) - 1}, or q to quit.')


def _select(title: str, options: list[str]) -> int:
    """Run curses menu, falling back to numbered input on failure."""
    try:
        return curses.wrapper(_curses_menu, title, options)
    except Exception:
        return _fallback_menu(title, options)


# ──────────────────────────────────────────────
# Public selection functions
# ──────────────────────────────────────────────

def select_camera() -> int:
    """
    Interactive camera-only selection.
    Returns the confirmed camera index, or exits the process if the user quits.
    """
    cameras = enumerate_cameras()
    if not cameras:
        print('No cameras detected — defaulting to index 0.')
        return 0

    while True:
        cam_labels = [f'Camera {i}  ({w}x{h})' for i, w, h in cameras]
        chosen = _select('Select camera input:', cam_labels)
        if chosen == _QUIT:
            sys.exit(0)
        idx, _, _ = cameras[chosen]
        if confirm_camera(idx):
            return idx


def run_device_selection() -> tuple[int, Optional[str]]:
    """
    Interactive camera + audio output selection.
    Returns (camera_index, audio_device_name_or_None), or exits on quit.
    """
    cameras = enumerate_cameras()
    if not cameras:
        print('No cameras detected — defaulting to index 0.')
        camera_index = 0
    else:
        while True:
            cam_labels = [f'Camera {i}  ({w}x{h})' for i, w, h in cameras]
            chosen = _select('Select camera input:', cam_labels)
            if chosen == _QUIT:
                sys.exit(0)
            idx, _, _ = cameras[chosen]
            if confirm_camera(idx):
                camera_index = idx
                break

    audio_outputs = enumerate_audio_outputs()
    audio_labels = ['Default (system output)'] + [name for _, name in audio_outputs]
    chosen = _select('Select audio output:', audio_labels)
    if chosen == _QUIT:
        sys.exit(0)
    audio_device = None if chosen == 0 else audio_outputs[chosen - 1][1]

    return camera_index, audio_device
