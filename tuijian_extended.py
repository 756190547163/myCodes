import tushare as ts
import pandas as pd

def fetch_top_stocks(ts_token, date, market_type=None):
    """
    获取沪深股通十大成交股
    """
    ts.set_token(ts_token)
    pro = ts.pro_api()
    try:
        params = {"trade_date": date}
        if market_type:
            params["market_type"] = market_type
        top_stocks = pro.hsgt_top10(**params)
        return top_stocks
    except Exception as e:
        print(f"获取沪深股通十大成交股失败: {e}")
        return None


if __name__ == "__main__":
    start_date = input("请输入日期（如 20220101）：")
    type = input("请输入市场类型（SH: 沪市, SZ: 深市）：")
    ts_token = "c652596d18e724a120b7fdedf85b2961b567712e4150ab68a214f2b7"  # 替换为你的 Tushare Token
    stocks = fetch_top_stocks(ts_token, start_date, market_type=type)
    for index, row in stocks.iterrows():
        print(f"{row['ts_code']} - {row['name']} - {row['close']}")
