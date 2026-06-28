import math


def _単位と除数(value: float) -> tuple[str, float]:
    v = abs(value)
    if v >= 100_000_000:
        return ("億円", 100_000_000.0)
    if v >= 10_000:
        return ("万円", 10_000.0)
    return ("円", 1.0)


def format_金額(value: float, *, with_unit: bool = True) -> str:
    """金額に応じた単位（億円/万円/円）を毎回決定して整形する。

    除数で割った値の絶対値が10未満のときだけ小数第1位まで表示し、
    10以上なら整数で表示する。with_unit=False なら単位を付けない。
    """
    label, divisor = _単位と除数(value)
    quotient = value / divisor
    base = f"{quotient:,.1f}" if abs(quotient) < 10 else f"{round(quotient):,}"
    return f"{base} {label}" if with_unit else base


def _inside_text_color(hex_color: str) -> str:
    """背景色のWCAG相対輝度から、白/濃色のうちコントラストが高い方の文字色を返す。"""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))

    def _lin(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    輝度 = 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)
    白コントラスト = (1.0 + 0.05) / (輝度 + 0.05)
    暗コントラスト = (輝度 + 0.05) / (0.04 + 0.05)
    return "#FFFFFF" if 白コントラスト >= 暗コントラスト else "#333333"


def _テーマ色() -> dict[str, str]:
    """Streamlit テーマに応じた背景・グリッド・テキスト色を返す。"""
    try:
        import streamlit as st
        is_dark = st.get_option("theme.base") == "dark"
    except Exception:
        is_dark = False
    if is_dark:
        return {
            "BG": "#0E1117",
            "PLOT": "#1C1E26",
            "GRID": "#3A3A4A",
            "TEXT": "#E0E0E0",
            "NEUTRAL": "#AAAAAA",  # ダーク背景でも見えるグレー
        }
    return {
        "BG": "#FFFFFF",
        "PLOT": "#F8F8F8",
        "GRID": "#DDDDDD",
        "TEXT": "#333333",
        "NEUTRAL": "#333333",
    }


def _nice_tick_interval(x_max: float) -> float:
    """x_max に対して 5〜6 本分の見やすい目盛り間隔を返す。"""
    if x_max <= 0:
        return 1.0
    raw = x_max / 5
    mag = 10 ** math.floor(math.log10(raw))
    n = raw / mag
    if n < 1.5:
        return mag
    if n < 3.5:
        return 2 * mag
    if n < 7.5:
        return 5 * mag
    return 10 * mag
