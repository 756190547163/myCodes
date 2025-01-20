import matplotlib.pyplot as plt
import pandas as pd
import tushare as ts
import json
from datetime import datetime

# 加载 Tushare 配置
with open("config/tushare_config.json") as f:
    config = json.load(f)

ts.set_token(config["tushare_token"])
pro = ts.pro_api()

# 获取股票数据并保存到 CSV 文件
def fetch_and_save_data(ts_code, start_date, end_date, file_path):
    """
    从 Tushare 获取数据并保存为 CSV 文件
    """
    print(f"获取股票 {ts_code} 的数据，从 {start_date} 到 {end_date}...")
    data = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    data.sort_values("trade_date", inplace=True)
    data.to_csv(file_path, index=False)
    print(f"数据已保存到 {file_path}")

# 计算均线与信号
def calculate_signals(data, fast_window=10, slow_window=20):
    """
    计算快均线、慢均线和交易信号
    """
    data["fast_ma"] = data["close"].rolling(fast_window).mean()
    data["slow_ma"] = data["close"].rolling(slow_window).mean()
    data["signal"] = 0
    data.loc[data["fast_ma"] > data["slow_ma"], "signal"] = 1  # 买入信号
    data.loc[data["fast_ma"] <= data["slow_ma"], "signal"] = -1  # 卖出信号
    return data

# 提取买卖信号
def extract_trade_signals(data):
    """
    提取买入和卖出信号的点
    """
    buy_signals = data[(data["signal"] == 1) & (data["signal"].shift(1) != 1)]
    sell_signals = data[(data["signal"] == -1) & (data["signal"].shift(1) != -1)]
    return buy_signals, sell_signals

# 绘图函数
def plot_backtest_result(data):
    """
    绘制回测结果，包括股票价格、均线和买卖信号
    """
    # 确保数据包含均线和信号列
    if "fast_ma" not in data.columns or "slow_ma" not in data.columns or "signal" not in data.columns:
        data = calculate_signals(data)

    # 提取买卖信号
    buy_signals, sell_signals = extract_trade_signals(data)

    plt.figure(figsize=(14, 7))

    # 绘制股票价格
    plt.plot(data["trade_date"], data["close"], label="Close Price", alpha=0.7)

    # 绘制快慢均线
    plt.plot(data["trade_date"], data["fast_ma"], label="Fast MA (10)", linestyle="--", alpha=0.7)
    plt.plot(data["trade_date"], data["slow_ma"], label="Slow MA (20)", linestyle="--", alpha=0.7)

    # 绘制买入信号
    plt.scatter(
        buy_signals["trade_date"],
        buy_signals["close"],
        label="Buy Signal",
        marker="^",
        color="green",
        alpha=1
    )

    # 绘制卖出信号
    plt.scatter(
        sell_signals["trade_date"],
        sell_signals["close"],
        label="Sell Signal",
        marker="v",
        color="red",
        alpha=1
    )

    plt.title("Backtest Result with Buy/Sell Signals")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 主程序
if __name__ == "__main__":
    # 用户输入股票代码和日期范围
    ts_code = input("请输入股票代码（如 000001.SZ）：")
    start_date = input("请输入开始日期（如 20230101）：")
    end_date = input("请输入结束日期（如 20231231）：")

    # 文件路径
    file_path = f"data/{ts_code.replace('.', '_')}.csv"

    # 下载数据
    fetch_and_save_data(ts_code, start_date, end_date, file_path)

    # 加载数据
    data = pd.read_csv(file_path)

    # 确保数据按日期排序
    data = data.sort_values(by="trade_date")

    # 绘制回测结果
    plot_backtest_result(data)
