"""
Interactive device selection for WaveSL.
Presents arrow-key menus in the terminal for camera and audio output selection.
A live camera preview confirms the correct device before the app starts.
"""

import curses
from typing import Optional

import cv2


# ──────────────────────────────────────────────
# Device enumeration
# ──────────────────────────────────────────────

def _camera_names_macos() -> list[str]:
    """
    Return camera display names in AVFoundation index order (matches OpenCV).
    Uses PyObjC AVFoundation if available; falls back to empty list so the
    caller shows generic labels instead of misleading mismatched names.
    """
    try:
        import AVFoundation
        devices = AVFoundation.AVCaptureDevice.devicesWithMediaType_(
            AVFoundation.AVMediaTypeVideo
        )
        return [d.localizedName() for d in devices]
    except Exception:
        return []


def enumerate_cameras() -> list[tuple[int, str, int, int]]:
    """
    Probe OpenCV indices 0–9 and return available cameras as
    [(index, name, width, height), ...].

    Names come from system_profiler (best-effort label; the live preview
    is the authoritative way to identify the correct device).
    """
    sys_names = _camera_names_macos()
    cameras: list[tuple[int, str, int, int]] = []

    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            name = sys_names[len(cameras)] if len(cameras) < len(sys_names) else f'Camera {i}'
            cameras.append((i, name, w, h))

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

def confirm_camera(index: int, name: str) -> bool:
    """
    Open a live preview window for the given camera index.
    Returns True if the user presses Enter (confirmed), False on Esc (go back).
    """
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f'  Could not open camera {index}.')
        return False

    window = 'WaveSL - Camera Preview'
    confirmed = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw name — black outline then white fill for readability on any background
        for color, thickness in [((0, 0, 0), 4), ((255, 255, 255), 2)]:
            cv2.putText(frame, name, (20, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, thickness, cv2.LINE_AA)

        # Draw instruction — black outline then yellow fill
        msg = 'ENTER = use this camera    ESC = go back'
        for color, thickness in [((0, 0, 0), 3), ((0, 220, 255), 2)]:
            cv2.putText(frame, msg, (20, 92),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, thickness, cv2.LINE_AA)

        cv2.imshow(window, frame)
        key = cv2.waitKey(30) & 0xFF
        if key in (10, 13):   # Enter
            confirmed = True
            break
        elif key == 27:        # Esc
            break

    cap.release()
    cv2.destroyAllWindows()
    return confirmed


# ──────────────────────────────────────────────
# Curses menu
# ──────────────────────────────────────────────

def _curses_menu(stdscr, title: str, options: list[str]) -> int:
    """Arrow-key curses menu. Returns the index of the selected option."""
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

        hint = 'UP/DOWN: navigate   ENTER: select'
        stdscr.addstr(h - 1, 0, hint[:w - 1])
        stdscr.refresh()

        key = stdscr.getch()
        if key == curses.KEY_UP and current > 0:
            current -= 1
        elif key == curses.KEY_DOWN and current < len(options) - 1:
            current += 1
        elif key in (curses.KEY_ENTER, 10, 13):
            return current


def _fallback_menu(title: str, options: list[str]) -> int:
    """Numbered fallback if curses is unavailable (e.g. IDE terminal)."""
    print(f'\n{title}')
    for i, opt in enumerate(options):
        print(f'  [{i}] {opt}')
    while True:
        try:
            val = int(input('Enter number: ').strip())
            if 0 <= val < len(options):
                return val
        except (ValueError, EOFError):
            pass
        print(f'  Enter a number between 0 and {len(options) - 1}.')


def _select(title: str, options: list[str]) -> int:
    """Run curses menu, falling back to numbered input on failure."""
    try:
        return curses.wrapper(_curses_menu, title, options)
    except Exception:
        return _fallback_menu(title, options)


# ──────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────

def select_camera() -> int:
    """
    Show interactive camera selection only.
    Returns the confirmed camera index.
    """
    cameras = enumerate_cameras()
    if not cameras:
        print('No cameras detected — defaulting to index 0.')
        return 0

    camera_index: Optional[int] = None
    while camera_index is None:
        cam_labels = [f'{name}  ({w}x{h})' for _, name, w, h in cameras]
        chosen = _select('Select camera input:', cam_labels)
        idx, name, _, _ = cameras[chosen]
        if confirm_camera(idx, name):
            camera_index = idx

    return camera_index


def run_device_selection() -> tuple[int, Optional[str]]:
    """
    Show interactive camera and audio output selection menus.
    Returns (camera_index, audio_device_name_or_None).
    """
    cameras = enumerate_cameras()
    if not cameras:
        print('No cameras detected — defaulting to index 0.')
        return 0, None

    audio_outputs = enumerate_audio_outputs()

    # Camera selection — loop until the user confirms the preview
    camera_index: Optional[int] = None
    while camera_index is None:
        cam_labels = [f'{name}  ({w}x{h})' for _, name, w, h in cameras]
        chosen = _select('Select camera input:', cam_labels)
        idx, name, _, _ = cameras[chosen]
        if confirm_camera(idx, name):
            camera_index = idx

    # Audio output selection
    audio_labels = ['Default (system output)'] + [name for _, name in audio_outputs]
    chosen = _select('Select audio output:', audio_labels)
    audio_device = None if chosen == 0 else audio_outputs[chosen - 1][1]

    return camera_index, audio_device
