import plotly.graph_objects as go
import streamlit as st
from loguru import logger

from chart_utils import _テーマ色, _単位と除数, format_金額


def build_breakeven_chart(
    *,
    売上高: float,
    支出: float,
    変動費率: float,
) -> go.Figure:
    """損益分岐点のmetric表示とグラフ生成を行う。

    売上線（y = x）と総費用線（y = 固定費 + 変動費率 × x）を描き、
    損失/利益ゾーンを色付きで塗りつぶす。metric表示も内部で行う。

    Parameters
    ----------
    売上高   : 現在の売上高
    支出     : 現在の総支出
    変動費率 : 売上高に対する変動費の割合（0 < 変動費率 < 1）
    """
    logger.debug("損益分岐点グラフ生成開始")

    テーマ = _テーマ色()
    _TEXT = テーマ["TEXT"]
    _BG = テーマ["BG"]
    _PLOT = テーマ["PLOT"]
    _GRID = テーマ["GRID"]
    _NEUTRAL = テーマ["NEUTRAL"]  # 損益分岐点マーカー・縦線色

    _GREEN = "#4caf50"
    _FONT_SIZE = 20
    _安全余裕率_目標 = 0.10

    変動費 = 売上高 * 変動費率
    固定費 = 支出 - 変動費
    限界利益率 = 1.0 - 変動費率

    if 限界利益率 <= 0:
        st.error("変動費率が1.00のため損益分岐点を計算できません。変動費率を下げてください。")
        st.stop()

    損益分岐点 = max(0.0, 固定費 / 限界利益率)
    安全余裕率目標達成売上 = max(0.0, 損益分岐点 / (1.0 - _安全余裕率_目標))
    安全余裕率現在 = (売上高 - 損益分岐点) / 売上高 if 売上高 > 0 else -1.0

    col1, col2, col3 = st.columns(3)
    col1.metric("損益分岐点売上高", format_金額(損益分岐点))

    if 売上高 >= 損益分岐点:
        col2.metric(
            "損益分岐点",
            "達成済み ✓",
            f"+{format_金額(売上高 - 損益分岐点)}",
        )
    else:
        col2.metric(
            "損益分岐点まで不足",
            format_金額(損益分岐点 - 売上高),
        )

    if 安全余裕率現在 >= _安全余裕率_目標:
        col3.metric(
            f"安全余裕率（目標 {_安全余裕率_目標:.0%}）",
            f"達成済み ✓  {安全余裕率現在:.1%}",
        )
    else:
        col3.metric(
            f"安全余裕率 {_安全余裕率_目標:.0%} まで不足",
            format_金額(安全余裕率目標達成売上 - 売上高),
        )

    x_max_raw = max(売上高, 安全余裕率目標達成売上, 損益分岐点) * 1.12
    label, divisor = _単位と除数(x_max_raw)
    x_max = x_max_raw / divisor

    n = 300
    xs = [x_max * i / (n - 1) for i in range(n)]
    bep_x = 損益分岐点 / divisor

    def _cost(x: float) -> float:
        return 固定費 / divisor + 変動費率 * x

    xs_loss = [x for x in xs if x <= bep_x]
    if not xs_loss or xs_loss[-1] < bep_x:
        xs_loss.append(bep_x)

    xs_profit = [x for x in xs if x >= bep_x]
    if not xs_profit or xs_profit[0] > bep_x:
        xs_profit.insert(0, bep_x)

    y_loss_cost = [_cost(x) for x in xs_loss]
    y_profit_cost = [_cost(x) for x in xs_profit]

    fig = go.Figure()

    if len(xs_loss) >= 2:
        fig.add_trace(go.Scatter(
            x=xs_loss + xs_loss[::-1],
            y=y_loss_cost + xs_loss[::-1],
            fill="toself",
            fillcolor="rgba(214, 39, 40, 0.12)",
            line=dict(width=0),
            name="損失ゾーン",
            hoverinfo="skip",
        ))

    if len(xs_profit) >= 2:
        fig.add_trace(go.Scatter(
            x=xs_profit + xs_profit[::-1],
            y=xs_profit + y_profit_cost[::-1],
            fill="toself",
            fillcolor="rgba(76, 175, 80, 0.15)",
            line=dict(width=0),
            name="利益ゾーン",
            hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter(
        x=xs,
        y=[_cost(x) for x in xs],
        mode="lines",
        name="総費用線",
        line=dict(color="#b45309", width=2.5),
    ))

    fig.add_trace(go.Scatter(
        x=xs,
        y=xs,
        mode="lines",
        name="売上線",
        line=dict(color=_GREEN, width=2.5),
    ))

    # 固定費の水平線＋注釈（プロット内左寄りに配置してクリップを回避）
    fig.add_shape(
        type="line",
        x0=0, x1=1,
        y0=固定費 / divisor, y1=固定費 / divisor,
        xref="paper", yref="y",
        line=dict(dash="dot", width=1.5, color="#60a5fa"),
        layer="below",
    )
    fig.add_annotation(
        x=0.02,
        y=固定費 / divisor,
        xref="paper", yref="y",
        text=f"固定費 {format_金額(固定費)}",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font=dict(size=_FONT_SIZE - 2, color="#60a5fa"),
        bgcolor=_PLOT,
    )

    for x_raw, color, symbol, name in (
        (損益分岐点, _NEUTRAL, "circle", f"損益分岐点  {format_金額(損益分岐点)}"),
        (売上高, _GREEN, "diamond", f"現在売上高  {format_金額(売上高)}"),
        (安全余裕率目標達成売上, "#7c3aed", "star", f"安全余裕率{_安全余裕率_目標:.0%}目標  {format_金額(安全余裕率目標達成売上)}"),
    ):
        xv = x_raw / divisor
        fig.add_vline(x=xv, line_dash="dot", line_width=1.5, line_color=color, layer="below")
        fig.add_trace(go.Scatter(
            x=[xv],
            y=[xv],
            mode="markers",
            name=name,
            marker=dict(color=color, size=14, symbol=symbol),
        ))

    x_tickformat = ",.1f" if x_max < 10 else ",.0f"

    fig.update_layout(
        title=dict(text="損益分岐点分析", font=dict(size=20, color=_TEXT)),
        paper_bgcolor=_BG,
        plot_bgcolor=_PLOT,
        font=dict(color=_TEXT, size=_FONT_SIZE),
        xaxis=dict(
            title=dict(text=f"売上高（{label}）", font=dict(size=_FONT_SIZE, color=_TEXT), standoff=5),
            range=[0, x_max],
            gridcolor=_GRID,
            zerolinecolor=_GRID,
            tickformat=x_tickformat,
            tickfont=dict(size=_FONT_SIZE, color=_TEXT),
        ),
        yaxis=dict(
            title=dict(text=f"金額（{label}）", font=dict(size=_FONT_SIZE, color=_TEXT), standoff=5),
            range=[0, x_max],
            gridcolor=_GRID,
            zerolinecolor=_GRID,
            tickformat=x_tickformat,
            tickfont=dict(size=_FONT_SIZE, color=_TEXT),
        ),
        legend=dict(
            orientation="h",
            y=-0.1,
            x=0,
            font=dict(color=_TEXT, size=_FONT_SIZE - 2),
            tracegroupgap=4,
        ),
        margin=dict(l=80, r=100, t=80, b=100),
        height=860,
    )
    logger.debug("損益分岐点グラフ生成完了")
    return fig
