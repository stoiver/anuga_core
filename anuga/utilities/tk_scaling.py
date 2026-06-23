"""HiDPI scaling helper for ANUGA's Tkinter GUIs.

On Wayland/XWayland sessions (and some misconfigured X servers) Tk reports a
bogus 96 DPI via ``winfo_fpixels('1i')`` -- and back-fills
``winfo_screenmmwidth`` to be consistent with it -- so estimating DPI from
either source also fails.  The result is that GUIs render physically tiny on
HiDPI panels.

``detect_dpi`` instead reads the user's *configured* DPI from the X resource
``Xft.dpi`` (the value GTK/Qt honour for HiDPI scaling, e.g. 192 -> 2x), and
``apply_hidpi_scaling`` uses it to enlarge Tk's named fonts so the whole UI
scales up.  On a normal 96 DPI display the scale resolves to 1.0 and nothing
changes, so this is safe to call unconditionally at startup.
"""

import os
import re
import subprocess
import tkinter as tk
import tkinter.font as tkfont


# Named fonts every Tk/ttk widget references by default; resizing these resizes
# the UI regardless of theme.
_NAMED_FONTS = ('TkDefaultFont', 'TkTextFont', 'TkFixedFont', 'TkMenuFont',
                'TkHeadingFont', 'TkCaptionFont', 'TkSmallCaptionFont',
                'TkIconFont', 'TkTooltipFont')


def _xft_dpi():
    """The user's configured Xft.dpi from ``xrdb -query``, or None."""
    try:
        out = subprocess.run(['xrdb', '-query'], capture_output=True,
                             text=True, timeout=2).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r'^Xft\.dpi:\s*([0-9.]+)', out, re.M)
    return float(m.group(1)) if m else None


def detect_dpi(root):
    """Best-effort *true* screen DPI, working around XWayland's bogus 96.

    Order: the user's configured Xft.dpi (preferred -- it reflects intended
    scaling, not raw panel density), then the QT_FONT_DPI env hint, then Tk's
    own (often bogus) numbers as a last resort.
    """
    dpi = _xft_dpi()
    if dpi and dpi > 0:
        return dpi
    try:
        v = float(os.environ.get('QT_FONT_DPI', ''))
        if v > 0:
            return v
    except ValueError:
        pass
    try:
        px = root.winfo_screenwidth()
        mm = root.winfo_screenmmwidth()
        tk_dpi = (px * 25.4 / mm) if mm and mm > 0 else 96.0
    except tk.TclError:
        tk_dpi = 96.0
    try:
        tk_dpi = max(tk_dpi, root.winfo_fpixels('1i'))
    except tk.TclError:
        pass
    return tk_dpi


def detect_scale(root, lo=1.0, hi=2.5):
    """Font/UI scale factor (1.0 == 96 DPI), clamped to [lo, hi]."""
    return round(min(hi, max(lo, detect_dpi(root) / 96.0)), 2)


def apply_hidpi_scaling(root, lo=1.0, hi=2.5):
    """Enlarge Tk's named fonts to suit the true screen DPI; return the scale.

    A no-op (returns 1.0) on normal-DPI displays, so it is safe to call
    unconditionally just after creating the root window.
    """
    scale = detect_scale(root, lo, hi)
    if scale <= 1.0:
        return scale
    for name in _NAMED_FONTS:
        try:
            f = tkfont.nametofont(name)
        except tk.TclError:
            continue
        size = f.cget('size')
        if not size:        # 0 == "unspecified"; leave it to Tk
            continue
        # Tk sizes: positive = points, negative = pixels. Preserve the sign.
        sign = 1 if size > 0 else -1
        f.configure(size=sign * max(8, int(round(abs(size) * scale))))
    return scale
