from statistics import mean

import pandas as pd
import streamlit as st

from inventory_chart.days_of_inventory import calc_doi


@st.cache_data(show_spinner=False)
def precompute_item_metrics(
    在庫推移データ: pd.DataFrame,
    COL_ITEM: str,
    COL_DATE: str,
    COL_OUTBOUND: str,
    COL_STOCK: str,
) -> pd.DataFrame:
    metric_rows: list[dict[str, str | float | None]] = []

    for 品番, 品番別データ in 在庫推移データ.groupby(COL_ITEM, sort=True):
        在庫数一覧 = 品番別データ[COL_STOCK].tolist()
        使用数一覧 = 品番別データ[COL_OUTBOUND].tolist()

        metric_rows.append(
            {
                COL_ITEM: str(品番),
                "日数(データ数)": round(float(len(品番別データ)), 1),
                "平均在庫数": round(mean(在庫数一覧), 1),
                "平均使用数": round(mean(使用数一覧), 1),
                "使用数分散": round(pd.Series(使用数一覧).var(), 1),
                "平均在庫日数（過去半年）": calc_doi(品番別データ, 180, COL_DATE, COL_OUTBOUND, COL_STOCK),
                "平均在庫日数（過去2年）": calc_doi(品番別データ, 730, COL_DATE, COL_OUTBOUND, COL_STOCK),
            }
        )

    if not metric_rows:
        return pd.DataFrame(columns=[COL_ITEM]).set_index(COL_ITEM)

    return pd.DataFrame(metric_rows).set_index(COL_ITEM)
