"""
Read a broker portfolio screenshot (Kite / Console / Groww / etc.) and extract
[Symbol, Buy Price, Qty] rows. Uses Tesseract OCR via pytesseract.

OCR is heuristic — always review the detected rows before adding them.

Setup:
    brew install tesseract          # the OCR engine (macOS)
    pip install pytesseract pillow
"""

import io
import re

import pandas as pd


def ocr_available():
    """(ok, message). Checks the pytesseract lib AND the tesseract binary."""
    try:
        import pytesseract  # noqa
        from PIL import Image  # noqa
    except Exception as e:
        return False, f"python libs missing ({e}); pip install pytesseract pillow"
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True, ""
    except Exception:
        return False, "tesseract engine not found — run: brew install tesseract"


def image_to_text(image_bytes) -> str:
    import pytesseract
    from PIL import Image
    img = Image.open(io.BytesIO(image_bytes)).convert("L")   # greyscale
    if img.width < 1500:                                     # upscale small shots
        scale = 1500 / img.width
        img = img.resize((int(img.width * scale), int(img.height * scale)))
    return pytesseract.image_to_string(img)


_NUM = re.compile(r"\d[\d,]*\.?\d*")
_SYM = re.compile(r"[A-Z][A-Z&\-]{1,18}")
_HEADER = {
    "INSTRUMENT", "QTY", "AVG", "COST", "LTP", "CUR", "VAL", "VALUE", "PNL",
    "NET", "CHG", "DAY", "HOLDINGS", "SYMBOL", "UNITS", "INVESTED", "CURRENT",
    "RETURNS", "STOCK", "SHARES", "PRICE", "TOTAL", "MARKET", "PORTFOLIO",
}


def parse_holdings(text: str) -> pd.DataFrame:
    """Heuristic: per line, take the leading UPPERCASE symbol, then the first
    number as Qty and the next as Buy Price (typical Kite/Console row order)."""
    rows = []
    for line in text.splitlines():
        s = line.strip()
        if len(s) < 3:
            continue
        m = _SYM.search(s.upper())
        if not m:
            continue
        sym = m.group(0).strip("-&")
        if sym in _HEADER or len(sym) < 2:
            continue
        nums = [float(x.replace(",", "")) for x in _NUM.findall(s[m.end():])]
        nums = [n for n in nums if n > 0]
        if len(nums) < 2:
            continue
        qty, buy = nums[0], nums[1]
        # Qty is usually a modest whole number; if the first looks like a price
        # and the second like a count, swap.
        if qty != int(qty) and buy == int(buy) and buy < 100000:
            qty, buy = buy, qty
        if qty <= 0 or buy <= 0 or qty > 1_000_000:
            continue
        rows.append({"Symbol": sym, "Buy Price": round(buy, 2), "Qty": int(round(qty))})
    df = pd.DataFrame(rows, columns=["Symbol", "Buy Price", "Qty"])
    if not df.empty:
        df = df.drop_duplicates(subset="Symbol", keep="first").reset_index(drop=True)
    return df
