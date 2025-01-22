# trading_system.py

import time
import tushare as ts
import pandas as pd
import json
import numpy as np
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from plyer import notification
from datetime import datetime, timedelta
from strategies.extended_strategy import ExtendedStrategy
from simulate_trading import AutoTrader

# 读取 Tushare 配置
with open("config/tushare_config.json") as f:
    config = json.load(f)

ts.set_token(config["tushare_token"])
pro = ts.pro_api()


def send_notification(title, message):
    """
    发送系统通知
    """
    notification.notify(
        title=title,
        message=message,
        app_name='交易系统',
        timeout=5  # 通知显示的时间（秒）
    )


def train_predict_model(data):
    """
    使用线性回归训练并预测第二天的股票信息
    """
    try:
        # 数据预处理
        data["trade_date"] = pd.to_datetime(data["trade_date"])
        data.sort_values("trade_date", inplace=True)
        data.bfill(inplace=True)

        features = ["open", "high", "low", "close", "vol", "ma5", "ma10", "vol_ratio"]
        target = ["open", "high", "low", "close"]

        X = data[features].values[:-1]
        y = data[target].values[1:]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LinearRegression()
        scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
        print(f"交叉验证 R^2 分数：{scores.mean():.4f} ± {scores.std():.4f}")

        model.fit(X_scaled, y)
        last_day_data = scaler.transform(data[features].iloc[-1].values.reshape(1, -1))
        predictions = model.predict(last_day_data)[0]
        predicted_open, predicted_high, predicted_low, predicted_close = predictions

        return {
            "predicted_open": float(predicted_open),
            "predicted_high": float(predicted_high),
            "predicted_low": float(predicted_low),
            "predicted_close": float(predicted_close)
        }
    except Exception as e:
        print(f"预测失败：{e}")
        return None


def fetch_historical_data(ts_code, start_date, end_date, input_year):
    """
    获取真实的股票历史数据
    """
    try:
        start_date3 = (datetime.now() - timedelta(days=int(input_year) * 365)).strftime('%Y%m%d')
        data = pro.daily(ts_code=ts_code, start_date=start_date3, end_date=end_date)
        if data.empty:
            print(f"无法获取 {ts_code} 的历史数据。")
            return None

        data = data[["trade_date", "open", "high", "low", "close", "vol"]]
        data.sort_values("trade_date", inplace=True)

        data["ma5"] = data["close"].rolling(window=5).mean()
        data["ma10"] = data["close"].rolling(window=10).mean()
        data["ma30"] = data["close"].rolling(window=30).mean()
        data["vol_ratio"] = data["vol"] / data["vol"].rolling(window=5).mean()
        return data
    except Exception as e:
        print(f"获取历史数据失败：{e}")
        return None


def fetch_realtime_data(ts_code, src="sina"):
    """
    获取实时数据
    """
    try:
        df = ts.realtime_quote(ts_code, src=src)
        if df.empty:
            print(f"无法获取实时数据，股票代码：{ts_code}")
            return None

        df.rename(
            columns={
                "TS_CODE": "ts_code",
                "PRICE": "close",
                "VOLUME": "vol",
                "DATE": "trade_date",
                "OPEN": "open",
                "HIGH": "high",
                "LOW": "low"
            },
            inplace=True
        )
        df["trade_date"] = datetime.now().strftime("%Y%m%d")
        return df[["ts_code", "open", "high", "low", "close", "vol", "trade_date"]]
    except Exception as e:
        print(f"获取实时数据失败：{e}")
        return None


def run_backtest_and_predict(ts_code, start_date, input_year, input_money, predict_option):
    """
    封装：执行历史回测 + (可选)预测
    返回回测日志(列表)和预测结果(可选)
    """
    end_date = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
    historical_data = fetch_historical_data(ts_code, start_date, end_date, input_year)
    if historical_data is None or historical_data.empty:
        return {"error": "无法获取历史数据"}, None

    # 初始化策略
    strategy = ExtendedStrategy(historical_data.copy())
    trader = AutoTrader(initial_balance=float(input_money))

    # 生成信号
    data_with_signals = strategy.generate_signals()
    sim_data = data_with_signals[data_with_signals["trade_date"] >= start_date]
    if sim_data.empty:
        return {"error": "指定日期区间无有效交易数据"}, None

    # 初始买入
    logs = []
    first_close = sim_data.iloc[0]["close"]
    trader.execute_trade(1, first_close)
    account_status = trader.get_account_status(first_close)  # 返回账户状态的字典
    logs.append({
        "trade_date": sim_data.iloc[0]["trade_date"],
        "action": "初始买入",
        "price": float(first_close),
        "account_status": account_status
    })

    # 模拟交易
    for _, row in sim_data.iterrows():
        day_log = {}
        day_log["trade_date"] = row["trade_date"]
        day_log["open"] = float(row["open"])
        day_log["high"] = float(row["high"])
        day_log["low"] = float(row["low"])
        day_log["close"] = float(row["close"])
        day_log["vol"] = float(row["vol"] * 100)

        # 获取交易信号
        signal = row["signal"]
        trader.execute_trade(signal, row["close"])

        # 获取并记录账户状态（现在是字典）
        acc_stat = trader.get_account_status(row["close"])
        day_log["signal"] = signal
        day_log["account_status"] = acc_stat

        logs.append(day_log)

    # (可选) 预测
    prediction_result = None
    if predict_option == "1":
        prediction_result = train_predict_model(strategy.data)

    return logs, prediction_result

def get_signal_message(latest_signal, ts_code, last_price):
    """
    根据信号返回一个用于前端展示的文字
    """
    if latest_signal == 1:
        return f"🔔 信号提示: 买入\n({ts_code}, 当前价: {last_price:.2f}) 🔔 "
    elif latest_signal == -1:
        return f"🔔 信号提示: 卖出\n({ts_code}, 当前价: {last_price:.2f}) 🔔 "
    else:
        return "🔔 信号提示: 观望 🔔 "

def get_realtime_signal(ts_code, historical_data, trader, strategy):
    """
    获取最新实时数据并判断买/卖/观望信号
    返回一个字典：包含实时价、信号、账户信息、以及 signal_message 等
    """
    # 这里 fetch_realtime_data(ts_code) ...
    realtime_data = fetch_realtime_data(ts_code)
    if realtime_data is None or realtime_data.empty:
        return {"error": "无法获取实时数据"}

    latest_row = realtime_data.iloc[-1]
    combined = pd.concat([historical_data, realtime_data], ignore_index=True)
    strategy.data = combined
    strategy.data = strategy.generate_signals()

    latest_signal = strategy.data.iloc[-1]["signal"]
    last_price = float(latest_row["close"])

    # 根据信号执行交易 + 记录提示
    if latest_signal == 1:
        print("\n===============================")
        print("      🔔 信号提示: 买入 🔔     ")
        print("===============================")
        send_notification("交易信号", f"{ts_code}买入信号: 当前价格 {last_price:.2f}")
        trader.execute_trade(latest_signal, last_price)
    elif latest_signal == -1:
        print("\n===============================")
        print("      🔔 信号提示: 卖出 🔔     ")
        print("===============================")
        send_notification("交易信号", f"{ts_code}卖出信号: 当前价格 {last_price:.2f}")
        trader.execute_trade(latest_signal, last_price)
    else:
        print("\n===============================")
        print("      🔔 信号提示: 观望 🔔     ")
        print("===============================")

    acc_stat = trader.get_account_status(last_price)

    return {
        "trade_date": latest_row["trade_date"],
        "current_time": datetime.now().strftime("%H:%M:%S"),
        "open": float(latest_row["open"]),
        "high": float(latest_row["high"]),
        "low": float(latest_row["low"]),
        "close": last_price,
        "vol": float(latest_row["vol"]),
        "signal": int(latest_signal),
        "signal_message": get_signal_message(latest_signal, ts_code, last_price),
        "account_status": acc_stat
    }