"""
pipeline/ocr.py – Telemetry extraction from drone HUD overlays.

Uses EasyOCR to read each frame's on-screen display and populate the
telemetry fields of the Frame dataclass.

Extracted fields
────────────────
  lat, lon          – GPS position (top-left green text)
  heading_deg       – Compass heading  (large centre-top number, 0-360°)
  alt_takeoff_ref_m – Height above takeoff point  (left value, bottom-right)
  alt_agl_m         – Height above ground (AGL)   (right value, bottom-right)

Both altitude values are parsed positionally (left = takeoff-ref, right = AGL),
not by magnitude, so the correct value is always assigned even when AGL exceeds
the takeoff-relative height (e.g. drone flying over terrain higher than its
launch point).

EasyOCR is initialised once and shared across all frames to avoid
reloading the neural-network weights on every call.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import cv2
import numpy as np

from pipeline.frame import Frame
import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level EasyOCR reader (lazy initialisation)
# ─────────────────────────────────────────────────────────────────────────────

_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        logger.info("Initialising EasyOCR (first call – may take a moment)…")
        _reader = easyocr.Reader(config.OCR_LANGUAGES, gpu=config.OCR_GPU)
    return _reader


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _crop(img: np.ndarray, window: tuple[int, int, int, int]) -> np.ndarray:
    """Return img[r0:r1, c0:c1] with boundary clamping."""
    r0, r1, c0, c1 = window
    h, w = img.shape[:2]
    return img[max(0, r0):min(h, r1), max(0, c0):min(w, c1)]


def _preprocess_lat_lon_crop(crop: np.ndarray) -> np.ndarray:
    """
    Enhance the lat/lon crop for better OCR accuracy.

    The HUD text is bright green on a dark background.  We:
      1. Extract the green channel to isolate the text.
      2. Upscale 3x so small characters (especially the decimal point)
         are large enough for EasyOCR to detect reliably.
      3. Apply CLAHE to boost local contrast.
    """
    green = crop[:, :, 1]                          # BGR → green channel
    h, w  = green.shape
    big   = cv2.resize(green, (w * 3, h * 3),      # 3× upscale
                       interpolation=cv2.INTER_CUBIC)
    clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(big)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)   # EasyOCR needs 3-ch


def _preprocess_altitude_crop(crop: np.ndarray) -> np.ndarray:
    """
    Enhance the altitude crop for better OCR accuracy.

    The altitude text is bright white on a dark semi-transparent background.
    We don't use channel extraction here — instead we:
      1. Convert to grayscale.
      2. Upscale 3x.
      3. Apply aggressive contrast stretching (normalize to full 0-255 range)
         so the faint decimal dot gets pulled up to the same brightness level
         as the surrounding digits.
      4. Apply CLAHE on top for local contrast boost.
      5. Threshold to sharpen edges — makes the dot a solid filled circle
         rather than a grey smudge.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    big  = cv2.resize(gray, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
    # Contrast stretch: pull the darkest pixel to 0, brightest to 255
    stretched = cv2.normalize(big, None, 0, 255, cv2.NORM_MINMAX)
    # CLAHE for local contrast
    clahe     = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    enhanced  = clahe.apply(stretched)
    # Invert so the dark decimal dot becomes a bright peak —
    # this ensures it survives any internal EasyOCR thresholding.
    inverted = cv2.bitwise_not(enhanced)
    return cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)


def _ocr_text(img_crop: np.ndarray) -> str:
    """Run EasyOCR on a crop and return concatenated raw text."""
    reader = _get_reader()
    results = reader.readtext(img_crop, detail=0, paragraph=False)
    return " ".join(results)


def _insert_decimal(digits: str, integer_digits: int) -> Optional[float]:
    """
    Reinsert a decimal point that OCR dropped.
    e.g. '32055649' with integer_digits=2 → 32.055649
    Only applied when the string is all digits and longer than integer_digits.
    """
    if digits.isdigit() and len(digits) > integer_digits:
        fixed = digits[:integer_digits] + "." + digits[integer_digits:]
        try:
            return float(fixed)
        except ValueError:
            pass
    return None


def _parse_coord(text: str, label: str, integer_digits: int) -> tuple[Optional[float], bool]:
    """
    Parse a single coordinate value after its label (LAT or LON).
    Handles both correct format (32.055649) and missing-decimal format (32055649).
    Label matching is intentionally loose to handle OCR artefacts like 'LATB'.

    Returns (value, used_decimal_fallback).
    """
    # Try with decimal point present
    match = re.search(rf"{label[:2]}[A-Z\s:.]*(-?\d{{1,3}}\.\d+)", text, re.IGNORECASE)
    if match:
        return float(match.group(1)), False

    # Try with decimal point missing — find the raw digit string after the label
    match = re.search(rf"{label[:2]}[A-Z\s:.]*(-?\d{{6,9}})", text, re.IGNORECASE)
    if match:
        value = _insert_decimal(match.group(1).lstrip("-"), integer_digits)
        return value, (value is not None)

    return None, False


def _parse_lat_lon(text: str) -> tuple[Optional[float], Optional[float], list[str]]:
    """
    Extract latitude and longitude from OCR text.

    Handles:
      - Normal:          'LAT: 32.055763  LON: 34.889088'
      - Missing decimal: 'LAT: 32055763   LON: 34889088'
      - Noisy label:     'LATB 32055763   LON: 34889088'
      - Comma as decimal: 'LAT: 32,055763  LON: 34,889088'

    Returns (lat, lon, fallbacks_used) where fallbacks_used lists which
    fields required the decimal-insertion fallback.
    """
    text = text.replace(',', '.')   # normalize comma-as-decimal-point
    lat, lat_fallback = _parse_coord(text, "LAT", integer_digits=2)
    lon, lon_fallback = _parse_coord(text, "LON", integer_digits=2)
    fallbacks = []
    if lat_fallback: fallbacks.append("lat")
    if lon_fallback: fallbacks.append("lon")
    return lat, lon, fallbacks


def _parse_heading(text: str) -> Optional[float]:
    """
    The heading is the largest standalone integer (0-360) in the crop.
    We pick the one with the most digits to avoid grabbing e.g. '86 m'
    distance values that might bleed into the crop.
    """
    candidates = re.findall(r"\b(\d{1,3})\b", text)
    valid = [int(c) for c in candidates if 0 <= int(c) <= 360]
    if not valid:
        return None
    # Return the value that is most likely a compass heading:
    # prefer 3-digit numbers; among ties take the first one found.
    three_digit = [v for v in valid if v >= 100]
    return float(three_digit[0]) if three_digit else float(valid[0])


def _fix_altitude_token(token: str) -> tuple[Optional[float], bool]:
    """
    Recover an altitude value from a digit string that is missing its decimal
    point — either because EasyOCR dropped it or misread it as '0'.

    Rules (altitudes in this dataset are always X.X or XX.X, max ~30 m):

      4 digits, 3rd digit is '0':  XX0X → XX.X   (dot misread as 0)
          e.g. '2103' → 21.3,  '1900' → 19.0
      3 digits, 2nd digit is '0':  X0X  → X.X    (dot misread as 0)
          e.g. '308'  → 3.8
      3 digits, 2nd digit is not '0':  XYZ → XY.Z  (dot simply dropped)
          e.g. '190'  → 19.0,  '136' → 13.6
      2 digits:  XY → X.Y                           (dot simply dropped)
          e.g. '17'   → 1.7

    Returns (value, used_fallback).
    """
    if not token.isdigit():
        return None, False

    n = len(token)
    fixed = None

    if n == 4 and token[2] == '0':
        fixed = float(token[:2] + '.' + token[3])       # XX0X → XX.X
    elif n == 3 and token[1] == '0':
        fixed = float(token[0] + '.' + token[2])        # X0X  → X.X
    elif n == 3:
        # Middle character is always the decimal point.
        # The OCR either dropped it silently or misread it as any character
        # (0, o, 5, 9, -, etc.). Either way: XYZ → X.Z
        # e.g. '956' → 9.6, '190' → 1.0, '308' → 3.8
        # If the real value were 95.6, OCR would give us '9506' (4 digits).
        fixed = float(token[0] + '.' + token[2])          # XYZ → X.Z
    elif n == 2:
        fixed = float(token[0] + '.' + token[1])        # XY   → X.Y

    if fixed is not None and 0.0 < fixed < 200.0:
        return fixed, True
    return None, False


def _parse_altitudes(text: str) -> tuple[Optional[float], Optional[float], list[str]]:
    """
    The bottom-right HUD shows TWO altitude readings left-to-right:
      1. Height above the takeoff/launch point  (barometric, left, down-arrow icon)
      2. Height above the ground right now      (radar/lidar AGL, right value)

    Returns (takeoff_ref, agl, fallbacks_used) where fallbacks_used lists
    the position labels ('takeoff_ref', 'agl') that required decimal fixing.

    Parsing strategy per token (left-to-right):
      - If it already contains a decimal point → parse directly.
      - If it is a 2–4 digit integer → apply _fix_altitude_token rules.
      - Anything else (letters, symbols) → skip.
    """
    values: list[float] = []
    fallbacks: list[str] = []
    position_labels = ['takeoff_ref', 'agl']

    text = text.replace(',', '.')   # normalize comma-as-decimal-point
    # Split on whitespace and process each token left-to-right
    for token in text.split():
        token = token.strip('.,;:')
        if not token:
            continue
        # Normalize common OCR digit confusions within numeric-looking tokens.
        # Only applied when the token contains at least one digit, to avoid
        # corrupting unit strings like 'KMH'.
        #   'o'/'O' → '0': e.g. '1o7' → '107' → format rule → '1.7'
        #   '-'     → '0': e.g. '14-5' → '1405' → format rule → '14.5'
        if any(c.isdigit() for c in token):
            token = token.replace('o', '0').replace('O', '0').replace('-', '0')

        # Already a proper decimal number
        if re.match(r'^\d{1,3}\.\d{1,2}$', token):
            try:
                v = float(token)
                if 0.0 < v < 200.0:
                    values.append(v)
                    continue
            except ValueError:
                pass

        # Integer token — try to recover the decimal point
        if token.isdigit() and 2 <= len(token) <= 4:
            fixed, used = _fix_altitude_token(token)
            if fixed is not None:
                values.append(fixed)
                # Record which position used the fallback
                idx = len(values) - 1
                if idx < len(position_labels):
                    fallbacks.append(position_labels[idx])

    if len(values) >= 2:
        return values[0], values[1], fallbacks
    elif len(values) == 1:
        return None, values[0], fallbacks
    return None, None, fallbacks


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_telemetry(frame: Frame) -> None:
    """
    Populate *frame* telemetry fields in-place by OCR-ing the raw image.
    Fields that cannot be parsed are left as None (a warning is logged).
    """
    img = frame.raw

    # ── LAT / LON ─────────────────────────────────────────────────────────────
    lat_lon_crop = _preprocess_lat_lon_crop(_crop(img, config.OCR_CROP_LAT_LON))
    lat_lon_text = _ocr_text(lat_lon_crop)
    logger.info("[%s] lat/lon OCR raw: %r", frame.stem, lat_lon_text)
    frame.lat, frame.lon, coord_fallbacks = _parse_lat_lon(lat_lon_text)
    if coord_fallbacks:
        logger.warning("[%s] Decimal point missing in OCR – used format fallback for: %s",
                       frame.stem, ", ".join(coord_fallbacks))

    # ── HEADING ───────────────────────────────────────────────────────────────
    heading_crop = _crop(img, config.OCR_CROP_HEADING)
    heading_text = _ocr_text(heading_crop)
    logger.debug("[%s] heading OCR: %r", frame.stem, heading_text)
    frame.heading_deg = _parse_heading(heading_text)

    # ── ALTITUDES ─────────────────────────────────────────────────────────────
    # Two readings parsed positionally: left = takeoff-ref, right = AGL.
    alt_crop = _preprocess_altitude_crop(_crop(img, config.OCR_CROP_ALT))
    alt_text = _ocr_text(alt_crop)
    logger.info("[%s] altitude OCR raw: %r", frame.stem, alt_text)
    frame.alt_takeoff_ref_m, frame.alt_agl_m, alt_fallbacks = _parse_altitudes(alt_text)
    if alt_fallbacks:
        logger.warning("[%s] Decimal point missing in altitude OCR – used format fallback for: %s",
                       frame.stem, ", ".join(alt_fallbacks))

    # ── Logging ───────────────────────────────────────────────────────────────
    ok  = frame.lat is not None and frame.lon is not None
    ok &= frame.heading_deg is not None
    ok &= frame.alt_takeoff_ref_m is not None

    if ok:
        terrain = frame.terrain_offset_m
        terrain_str = f"  terrain_offset={terrain:+.1f}m" if terrain is not None else ""
        logger.info(
            "[%s] OCR OK – lat=%.6f  lon=%.6f  hdg=%.0f°  "
            "alt_ref=%.1f m  agl=%.1f m%s",
            frame.stem, frame.lat, frame.lon, frame.heading_deg,
            frame.alt_takeoff_ref_m,
            frame.alt_agl_m if frame.alt_agl_m is not None else float("nan"),
            terrain_str,
        )
    else:
        missing = [
            fname for fname, val in [
                ("lat",          frame.lat),
                ("lon",          frame.lon),
                ("heading",      frame.heading_deg),
                ("alt_takeoff",  frame.alt_takeoff_ref_m),
            ] if val is None
        ]
        logger.warning("[%s] OCR failed for: %s", frame.stem, ", ".join(missing) or "(unknown field)")


def extract_telemetry_all(frames: list[Frame]) -> None:
    """Run extract_telemetry on every frame in *frames*."""
    for i, frame in enumerate(frames):
        logger.info("OCR %d/%d: %s", i + 1, len(frames), frame.stem)
        extract_telemetry(frame)
