import plotly.graph_objects as go
from loguru import logger

from chart_utils import (
    _inside_text_color,
    _nice_tick_interval,
    _テーマ色,
    _単位と除数,
    format_金額,
)


def build_summary_chart(
    *,
    売上金額: float,
    材料費: float,
    外注費: float,
    社内加工費: float,
    間接費: float,
    販管費: float,
    営業外費用: float,
    特別損失: float,
    法人税: float,
) -> go.Figure:
    """売上・費用内訳・粗利・純利を共通横軸の4行で表示する横棒グラフ。

    行（上から）:
      1. 売上（0起点の単棒）
      2. 支出（8項目を左から積み上げ）
      3. 粗利（= 売上 − 製造原価）
      4. 純利（= 売上 − 支出）

    粗利・純利のバーは右端を x=売上 に固定し、プラスなら
    そこから左へ緑、マイナスなら売上から右へ赤で描く。
    """
    logger.debug("サマリーグラフ生成開始")

    テーマ = _テーマ色()
    _TEXT = テーマ["TEXT"]
    _BG = テーマ["BG"]
    _PLOT = テーマ["PLOT"]
    _GRID = テーマ["GRID"]

    _GREEN = "#4caf50"
    _行高_px = 130
    _上余白_px = 100
    _下余白_px = 90
    _FONT_SIZE = 20
    _狭幅閾値 = 0.04

    _ROW_売上 = "売上"
    _ROW_支出 = "支出"
    _ROW_粗利 = "粗利"
    _ROW_純利 = "純利"

    _LIGHT_GREEN = "#66bb6a"
    _DARK_GREEN = "#2a8c4a"
    _RED = "#d62728"

    _COST_COLORS: dict[str, str] = {
        "材料費": "#cbd5e8",
        "外注費": "#2563eb",
        "社内加工費": "#3b82f6",
        "間接費": "#60a5fa",
        "販管費": "#fbbf24",
        "営業外費用": "#f59e0b",
        "特別損失": "#d97706",
        "法人税": "#c2410c",
    }

    製造原価 = 材料費 + 外注費 + 社内加工費 + 間接費
    支出 = 製造原価 + 販管費 + 営業外費用 + 特別損失 + 法人税
    粗利 = 売上金額 - 製造原価
    純利 = 売上金額 - 支出

    label, divisor = _単位と除数(max(売上金額, 支出))

    # x_上限: autorange の代わりにvline幅・狭幅判定に使用
    右端 = max(売上金額, 支出) / divisor
    x_上限 = 右端 * 1.15 if 右端 > 0 else 1.0
    x_tickformat = ",.1f" if x_上限 < 10 else ",.0f"
    dtick = _nice_tick_interval(x_上限)

    fig = go.Figure()

    def _add_value_label(row: str, x_right: float, text: str, color: str) -> None:
        fig.add_annotation(
            x=x_right,
            y=row,
            xref="x",
            yref="y",
            text=text,
            showarrow=False,
            xanchor="left",
            xshift=6,
            font=dict(size=_FONT_SIZE, color=color),
            bgcolor=_PLOT,
        )

    cost_items = [
        ("材料費", 材料費),
        ("外注費", 外注費),
        ("社内加工費", 社内加工費),
        ("間接費", 間接費),
        ("販管費", 販管費),
        ("営業外費用", 営業外費用),
        ("特別損失", 特別損失),
        ("法人税", 法人税),
    ]
    累計 = 0.0
    for name, val in cost_items:
        fig.add_trace(
            go.Bar(
                name=name,
                y=[_ROW_支出],
                x=[val / divisor],
                base=[累計 / divisor],
                orientation="h",
                marker_color=_COST_COLORS[name],
                marker_line_width=0,
                text=[
                    format_金額(val, with_unit=False)
                    if (val / divisor) / x_上限 >= _狭幅閾値
                    else ""
                ],
                textposition="inside",
                insidetextanchor="middle",
                constraintext="inside",
                textfont=dict(size=_FONT_SIZE, color=_inside_text_color(_COST_COLORS[name])),
                hovertemplate=f"{name}: {format_金額(val)}<extra></extra>",
            )
        )
        累計 += val
    _add_value_label(_ROW_支出, 支出 / divisor, format_金額(支出), "#b45309")

    fig.add_trace(
        go.Bar(
            name=_ROW_売上,
            y=[_ROW_売上],
            x=[売上金額 / divisor],
            base=[0.0],
            orientation="h",
            marker_color=_LIGHT_GREEN,
            marker_line_width=0,
            showlegend=False,
            hovertemplate=f"売上: {format_金額(売上金額)}<extra></extra>",
        )
    )
    _add_value_label(_ROW_売上, 売上金額 / divisor, format_金額(売上金額), _LIGHT_GREEN)

    def _add_profit_bar(row: str, profit: float) -> None:
        if profit >= 0:
            base = (売上金額 - profit) / divisor
            width = profit / divisor
            color = _GREEN if row == _ROW_粗利 else _DARK_GREEN
            text_color = color
        else:
            base = 売上金額 / divisor
            width = -profit / divisor
            color = "#e57373" if row == _ROW_粗利 else _RED
            text_color = color
        fig.add_trace(
            go.Bar(
                name=row,
                y=[row],
                x=[width],
                base=[base],
                orientation="h",
                marker_color=color,
                marker_line_width=0,
                showlegend=False,
                hovertemplate=f"{row}: {format_金額(profit)}<extra></extra>",
            )
        )
        _add_value_label(row, base + width, format_金額(profit), text_color)

    _add_profit_bar(_ROW_粗利, 粗利)
    _add_profit_bar(_ROW_純利, 純利)

    # 等間隔の縦グリッド線（バー間の空白に現れる）
    tick_x = 0.0
    while tick_x <= x_上限 * 1.001:
        fig.add_vline(x=tick_x, line_color=_GRID, line_width=1, layer="below")
        tick_x += dtick

    # 参照ライン: 売上・原価・支出の位置を縦点線で示す
    for x_val, line_color, label_text, y_paper in (
        (売上金額 / divisor, _LIGHT_GREEN, "売上", 0.505),
        (製造原価 / divisor, "#2563eb", "原価", 1.015),
        (支出 / divisor, "#b45309", "支出", 0.785),
    ):
        fig.add_vline(
            x=x_val, line_dash="dot", line_width=1.5, line_color=line_color, layer="below"
        )
        fig.add_annotation(
            x=x_val,
            y=y_paper,
            xref="x",
            yref="paper",
            text=label_text,
            showarrow=False,
            font=dict(size=_FONT_SIZE, color=line_color),
            bgcolor=_PLOT,
        )

    行ラベル = [_ROW_支出, _ROW_売上, _ROW_粗利, _ROW_純利]
    チャート高さ = _上余白_px + _下余白_px + _行高_px * len(行ラベル)

    fig.update_layout(
        barmode="overlay",
        bargap=0.45,
        title=dict(
            text="売上 / 費用内訳 / 粗利 / 純利（共通横軸）",
            font=dict(size=20, color=_TEXT),
        ),
        paper_bgcolor=_BG,
        plot_bgcolor=_PLOT,
        font=dict(color=_TEXT, size=_FONT_SIZE),
        xaxis=dict(
            autorange=True,
            tickformat=x_tickformat,
            dtick=dtick,
            tick0=0,
            title=dict(text=f"（{label}）", font=dict(size=_FONT_SIZE - 4, color=_TEXT), standoff=5),
            gridcolor=_GRID,
            zerolinecolor=_GRID,
            tickfont=dict(size=_FONT_SIZE, color=_TEXT),
        ),
        yaxis=dict(
            categoryorder="array",
            categoryarray=行ラベル,
            autorange="reversed",
            tickfont=dict(size=_FONT_SIZE, color=_TEXT),
        ),
        legend=dict(
            orientation="h",
            y=-0.18,
            x=0,
            font=dict(color=_TEXT, size=_FONT_SIZE),
            traceorder="normal",
        ),
        margin=dict(l=70, r=100, t=_上余白_px, b=_下余白_px),
        height=チャート高さ,
    )
    logger.debug("サマリーグラフ生成完了")
    return fig
