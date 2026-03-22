from datetime import datetime
from datetime import timedelta

import pandas as pd

def calc_doi(df, days, COL_DATE, COL_OUTBOUND, COL_STOCK):
    """
    end_date を基準に、過去 days 日間の在庫日数を計算する。
    
    在庫日数 = 期間内の平均在庫数 ÷ (期間内の1日平均出荷数)
    """
    date_series = pd.to_datetime(df[COL_DATE])
    end_date = pd.Timestamp(datetime.now()).normalize()
    start_date = end_date - timedelta(days=days)

    period = df[(date_series > start_date) & (date_series <= end_date)]
    
    if period.empty:
        return None
    
    # 期間内の平均在庫数
    avg_inventory = period[COL_STOCK].mean()
    
    # 期間中の1日平均出荷数
    avg_daily_shipment = period[COL_OUTBOUND].sum() / days
    
    if avg_daily_shipment == 0:
        return float("inf")
    
    doi = avg_inventory / avg_daily_shipment
    return round(doi, 1)