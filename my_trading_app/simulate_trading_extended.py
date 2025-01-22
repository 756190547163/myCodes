# -*- coding: utf-8 -*-

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

# 加载 Tushare 配置
with open("config/tushare_config.json") as f:
    config = json.load(f)

ts.set_token(config["tushare_token"])
pro = ts.pro_api()


# 初始化系统通知
def send_notification(title, message):
    """
    发送系统通知
    """
    notification.notify(
        title=title,
        message=message,
        app_name='交易系统',
        app_icon=None,
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

        # 填充空值（均线和成交量比可能存在空值）
        data.bfill(inplace=True)

        # 选取特征和目标
        features = ["open", "high", "low", "close", "vol", "ma5", "ma10", "vol_ratio"]
        target = ["open", "high", "low", "close"]

        # 构造训练数据
        X = data[features].values[:-1]
        y = data[target].values[1:]

        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 模型训练和交叉验证
        model = LinearRegression()
        scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
        print(f"交叉验证 R^2 分数：{scores.mean():.4f} ± {scores.std():.4f}")

        model.fit(X_scaled, y)

        # 使用最后一天的数据预测第二天的数据
        last_day_data = scaler.transform(data[features].iloc[-1].values.reshape(1, -1))
        predictions = model.predict(last_day_data)[0]

        # 构造预测结果
        predicted_open, predicted_high, predicted_low, predicted_close = predictions

        return {
            "predicted_open": predicted_open,
            "predicted_high": predicted_high,
            "predicted_low": predicted_low,
            "predicted_close": predicted_close
        }
    except Exception as e:
        print(f"预测失败：{e}")
        return None


def fetch_historical_data(ts_code, start_date, end_date, input_year):
    """
    获取真实的股票历史数据
    """
    try:
        # 计算开始日期，通过入参 input_year 向前回溯三年
       # start_date3 = (datetime.now() - timedelta(days=  * 365)).strftime('%Y%m%d')
        start_date3 = (datetime.now() - timedelta(days=int(input_year) * 365)).strftime('%Y%m%d')
        # 调用 Tushare 的 daily 接口获取日线数据
        data = pro.daily(ts_code=ts_code, start_date=start_date3, end_date=end_date)
        if data.empty:
            print(f"无法获取 {ts_code} 的历史数据。")
            return None

        # 保留必要列，并确保数据按照日期排序
        data = data[["trade_date", "open", "high", "low", "close", "vol"]]
        data.sort_values("trade_date", inplace=True)

        # 计算均线和成交量比
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

        # 格式化实时数据
        df.rename(columns={"TS_CODE": "ts_code", "PRICE": "close", "VOLUME": "vol", "DATE": "trade_date", "OPEN": "open", "HIGH": "high", "LOW": "low"}, inplace=True)
        df["trade_date"] = datetime.now().strftime("%Y%m%d")
        return df[["ts_code", "open", "high", "low", "close", "vol", "trade_date"]]
    except Exception as e:
        print(f"获取实时数据失败：{e}")
        return None

if __name__ == "__main__":
    ts_code = input("请输入股票代码（如 000001.SZ）：")
    start_date = input("请输入投资开始日期（如 20220101）：")
    input_year = input("请输入策略分析年数（如输入3则为当前时间倒退3年为分析开始时间）：")
    input_money = input("请输入初始资金（如 100000.00）：")
    predict_option = input("是否预测第二天的股票信息？（1: 是, 0: 否）：")

    # 获取真实的历史数据
    end_date = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
    historical_data = fetch_historical_data(ts_code, start_date, end_date, input_year)
    if historical_data is None or historical_data.empty:
        print("无法获取历史数据，程序退出。")
        exit()

    # 初始化策略和交易器
    strategy = ExtendedStrategy(historical_data.copy())
    trader = AutoTrader(initial_balance=float(input_money))

    # 模拟交易历史数据
    # 计算综合信号
    data_with_signals = strategy.generate_signals()
    # 筛选模拟交易范围内的数据
    sim_data = data_with_signals[data_with_signals["trade_date"] >= start_date]

    # 初始买入操作
    initial_price = sim_data.iloc[0]['close']  # 使用第一个交易日的收盘价
    print("\n初始买入操作：")
    trader.execute_trade(1, initial_price)
    trader.get_account_status(initial_price)

    for index, row in sim_data.iterrows():
        print(f"\n日期：{row['trade_date']}")
        print(f"开盘价：{row['open']:.2f}，最高价：{row['high']:.2f}，最低价：{row['low']:.2f}，收盘价：{row['close']:.2f}，成交量：{row['vol']*100:.0f}")

        # 获取当天信号
        signal = row["signal"]

        # 执行交易
        trader.execute_trade(signal, row["close"])

        # 输出账户状态，包括浮盈浮亏
        trader.get_account_status(row["close"])

    # 如果用户选择预测
    if predict_option == "1":
        prediction = train_predict_model(strategy.data)
        if prediction:
            print("\n预测股票信息：")
            print(f"预测开盘价：{prediction['predicted_open']:.2f}")
            print(f"预测最高价：{prediction['predicted_high']:.2f}")
            print(f"预测最低价：{prediction['predicted_low']:.2f}")
            print(f"预测收盘价：{prediction['predicted_close']:.2f}")
            # 可视化预测结果
#             visualize_predictions(historical_data, prediction)
    # 开始实时数据
    print("开始实时数据刷新...")
    while True:
        # 获取最新实时数据
        realtime_data = fetch_realtime_data(ts_code)
        if realtime_data is not None:
            latest_row = realtime_data.iloc[-1]
            print(f"\n实时数据: 日期: {latest_row['trade_date']}, 时间: {datetime.now().strftime('%H:%M:%S')} ")
            print(f"开盘价: {latest_row['open']:.2f}, 最高价: {latest_row['high']:.2f}, 最低价: {latest_row['low']:.2f}, 当前价: {latest_row['close']:.2f}, 成交量: {latest_row['vol']:.0f}")

            # is_realtime 赋值
            strategy.is_realtime = True
            # 将实时数据添加到历史数据中
            historical_datac = pd.concat([historical_data, realtime_data], ignore_index=True)

            # 结合策略计算信号
            strategy.data = historical_datac
            strategy.data = strategy.generate_signals()
            latest_signal = strategy.data.iloc[-1]["signal"]
            if latest_signal == 1:
                print("\n===============================")
                print("      🔔 信号提示: 买入 🔔     ")
                print("===============================")
                send_notification("交易信号", f"{ts_code}买入信号: 当前价格 {latest_row['close']:.2f}")
                trader.execute_trade(latest_signal, latest_row['close'])
            elif latest_signal == -1:
                print("\n===============================")
                print("      🔔 信号提示: 卖出 🔔     ")
                print("===============================")
                send_notification("交易信号", f"{ts_code}卖出信号: 当前价格 {latest_row['close']:.2f}")
                trader.execute_trade(latest_signal, latest_row['close'])
            else:
                print("\n===============================")
                print("      🔔 信号提示: 观望 🔔     ")
                print("===============================")
            # 输出账户状态，包括浮盈浮亏
            trader.get_account_status(latest_row['close'])

        # 每5秒刷新一次
        time.sleep(10)
