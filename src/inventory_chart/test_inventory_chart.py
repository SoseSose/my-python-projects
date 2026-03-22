import random
import tempfile
import os
from datetime import date, timedelta

import pandas as pd

from inventory_chart.inventory_chart import make_dashboard

INITIAL_STOCK = 500
LOOKBACK_DAYS = 720
ITEMS = ["A001", "A002", "A003", "A004", "A005"]

COL_ITEM = "品番"
COL_DATE = "日付"
COL_INBOUND = "補充数"
COL_OUTBOUND = "使用数"
COL_FINAL_STOCK = "最終在庫数"


def 入出庫データを生成() -> pd.DataFrame:
    random.seed(42)
    開始日 = date.today() - timedelta(days=LOOKBACK_DAYS)
    終了日 = date.today()

    rows: list[dict[str, str | date | int]] = []
    品番別在庫数 = {品番: INITIAL_STOCK + index * 80 for index, 品番 in enumerate(ITEMS)}

    経過日数 = (終了日 - 開始日).days
    for day in range(経過日数 + 1):
        当日 = 開始日 + timedelta(days=day)

        for index, 品番 in enumerate(ITEMS):
            在庫数 = 品番別在庫数[品番]
            補充周期 = 5 + index
            補充数 = random.randint(140 + index * 10, 240 + index * 10) if day % 補充周期 == 0 else 0
            使用要求数 = random.randint(18 + index * 2, 42 + index * 2)
            使用数 = min(使用要求数, 在庫数 + 補充数)
            在庫数 = 在庫数 + 補充数 - 使用数
            品番別在庫数[品番] = 在庫数

            rows.append(
                {
                    COL_ITEM: 品番,
                    COL_DATE: 当日,
                    COL_OUTBOUND: 使用数,
                    COL_INBOUND: 補充数,
                    COL_FINAL_STOCK: 在庫数,
                }
            )

    return pd.DataFrame(rows)


入出庫データ = 入出庫データを生成()

_tmp_dir = tempfile.mkdtemp()
_csv_path = os.path.join(_tmp_dir, "inventory_data.csv")
入出庫データ.to_csv(_csv_path, index=False, encoding="utf-8-sig")

make_dashboard(
    _csv_path,
    COL_ITEM,
    COL_DATE,
    COL_INBOUND,
    COL_OUTBOUND,
    COL_FINAL_STOCK,
)



