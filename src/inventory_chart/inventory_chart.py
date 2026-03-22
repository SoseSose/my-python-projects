from typing import cast

import pandas as pd

import streamlit as st
from inventory_chart.item_metrics import precompute_item_metrics
from inventory_chart.plot_settings import make_stock_chart, make_change_chart, usage_analysis


def sort_by_date(data: pd.DataFrame, COL_DATE: str) -> pd.DataFrame:
    並び順インデックス = pd.to_datetime(data[COL_DATE]).sort_values().index
    return cast(pd.DataFrame, data.loc[並び順インデックス].copy())


def make_dashboard(
    csv_path: str,
    COL_ITEM: str,
    COL_DATE: str,
    COL_INBOUND: str,
    COL_OUTBOUND: str,
    COL_STOCK: str,
) -> None:
    #品番を選択して、その品番のデータから算出するように変更する。
    st.set_page_config(page_title="在庫推移可視化", layout="wide")
    st.title("在庫推移可視化")

    在庫推移データ = pd.read_csv(csv_path)
    事前計算済metric = precompute_item_metrics(在庫推移データ, COL_ITEM, COL_DATE, COL_OUTBOUND, COL_STOCK)
    品番一覧 = 事前計算済metric.index.tolist()
    if not 品番一覧:
        st.info("表示できるデータがありません。")
        return

    選択品番 = st.radio("品番を選択", 品番一覧, horizontal=True)
    表示対象データ = 在庫推移データ[在庫推移データ[COL_ITEM] == 選択品番].copy()
    表示対象データ = sort_by_date(表示対象データ, COL_DATE)
    選択品番metric =  事前計算済metric.loc[str(選択品番)]


    st.subheader("在庫データの要約")

    show_metric = 選択品番metric.to_dict()
    cols = st.columns(len(show_metric))
    for c, key in zip(cols, show_metric.keys()):
        c.metric(str(key), show_metric[key])

    csv_data = 事前計算済metric.to_csv(index=True, encoding="utf-8-sig")
    st.download_button(
        label="メトリクスをCSVでダウンロード",
        data=csv_data,
        file_name="item_metrics.csv",
        mime="text/csv",
    )
    #直近半年の回転区分、１年の回転区分、２年の回転区分を算出

    stock_fig = make_stock_chart(表示対象データ, COL_DATE, COL_OUTBOUND, COL_STOCK)
    change_fig = make_change_chart(表示対象データ, COL_DATE, COL_INBOUND, COL_OUTBOUND)
    st.plotly_chart(stock_fig, width="stretch")
    st.plotly_chart(change_fig, width="stretch")

    st.plotly_chart(usage_analysis(表示対象データ, COL_DATE, COL_OUTBOUND), width="stretch")

    

    with st.expander("データを表示"):
        st.dataframe(表示対象データ, width="stretch")
