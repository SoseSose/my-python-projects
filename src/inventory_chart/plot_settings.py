import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import pandas as pd
from collections import Counter
from inventory_chart.safety_stock import get_safety_stock

def make_stock_chart(
    data: pd.DataFrame,
    COL_DATE: str,
    COL_OUTBOUND: str,
    COL_STOCK: str,
) -> go.Figure:
    if data.empty:
        return go.Figure()


    # 出庫分（負の増減の絶対値）を安全在庫計算に使用
    出庫一覧 = pd.Series(data[COL_OUTBOUND].tolist())
    safety_stock = get_safety_stock(出庫一覧, safety_factor=0.95) if len(出庫一覧) > 0 else 0

    日付一覧 = data[COL_DATE].tolist()
    在庫数一覧 = data[COL_STOCK].tolist()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=日付一覧,
            y=在庫数一覧,
            mode="lines+markers",
            name="在庫数",
            line={"color": "steelblue", "width": 2},
            marker={"size": 5},
        )
    )

    fig.add_shape(
        type="line",
        xref="paper",
        yref="y",
        x0=0,
        x1=1,
        y0=safety_stock,
        y1=safety_stock,
        line={"color": "orange", "width": 2, "dash": "dash"},
    )
    fig.add_annotation(
        xref="paper",
        yref="y",
        x=1,
        y=safety_stock,
        text=f"安全在庫 {safety_stock}",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        yshift=6,
        font={"color": "orange"},
        bgcolor="rgba(255,255,255,0.7)",
    )

    fig.update_xaxes(title_text="日付", tickformat="%y/%m/%d", hoverformat="%y/%m/%d", tickangle=-35)

    stock_max = math.ceil(max(在庫数一覧) * 1.05)
    fig.update_yaxes(title_text="在庫数", range=[0, stock_max])

    fig.update_layout(height=420)
    return fig


def make_change_chart(
    data: pd.DataFrame,
    COL_DATE: str,
    COL_INBOUND: str,
    COL_OUTBOUND: str,
) -> go.Figure:
    if data.empty:
        return go.Figure()

    日付一覧 = data[COL_DATE].tolist()
    入庫一覧 = data[COL_INBOUND].tolist()
    出庫一覧 = data[COL_OUTBOUND].tolist()
    出庫表示一覧 = [-value for value in 出庫一覧]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=日付一覧,
            y=入庫一覧,
            marker_color="#2ca02c",
            name="入庫数",
        )
    )
    fig.add_trace(
        go.Bar(
            x=日付一覧,
            y=出庫表示一覧,
            marker_color="#d62728",
            name="出庫数",
            customdata=出庫一覧,
            hovertemplate="日付=%{x|%y/%m/%d}<br>出庫数=%{customdata}<extra></extra>",
        )
    )

    fig.update_xaxes(title_text="日付", tickformat="%y/%m/%d", hoverformat="%y/%m/%d", tickangle=-35)
    fig.update_yaxes(title_text="数量")

    max_value = max(max(入庫一覧, default=0), max(出庫一覧, default=0))
    axis_limit = math.ceil(max_value * 1.05) if max_value > 0 else 1
    fig.update_yaxes(range=[-axis_limit, axis_limit], zeroline=True, zerolinewidth=2, zerolinecolor="#666")

    fig.update_layout(bargap=0.05, barmode="relative", height=320)
    return fig

def usage_analysis(data: pd.DataFrame, COL_DATE: str, COL_OUTBOUND: str)  -> go.Figure:
    if data.empty:
        return go.Figure()

    日付一覧 = data[COL_DATE].tolist()
    出庫数一覧 = data[COL_OUTBOUND].tolist()


    min_x = min(出庫数一覧)
    max_x = max(出庫数一覧)
    counts = Counter(出庫数一覧)
    bin_values = list(range(min_x, max_x + 1))
    bin_counts = [counts.get(x, 0) for x in bin_values]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("出庫量ヒストグラム（時系列）", "出庫分布（90度回転）"),
        horizontal_spacing=0.12,
    )

    fig.add_trace(
        go.Bar(
            x=日付一覧,
            y=出庫数一覧,
            marker_color="#d62728",
            name="出庫量",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=bin_counts,
            y=bin_values,
            orientation="h",
            marker_color="#f8870e",
            name="頻度",
        ),
        row=1,
        col=2,
    )


    fig.update_xaxes(
        title_text="日付",
        tickformat="%y/%m/%d",
        hoverformat="%y/%m/%d",
        tickangle=-35,
        row=1,
        col=1,
    )
    y_min = -1
    y_max = math.ceil(1.03 * max_x)

    fig.update_yaxes(title_text="1日あたりの出庫数", range=[y_min, y_max], row=1, col=1)

    fig.update_xaxes(title_text="頻度", tickformat="d", row=1, col=2)
    fig.update_yaxes(
        title_text="1日あたりの出庫数",
        range=[y_min, y_max],
        tickformat="d",
        row=1,
        col=2,
    )

    fig.update_layout(
        bargap=0.05,
        showlegend=False,
    )
    return fig