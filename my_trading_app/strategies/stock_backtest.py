import matplotlib.pyplot as plt
import pandas as pd
import tushare as ts
import json
from ma_strategy import MaStockStrategy
from rsi_strategy import RsiStockStrategy
from strategy_manager import StrategyManager
from matplotlib import rcParams


# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

# 加载 Tushare 配置
with open("config/tushare_config.json") as f:
    config = json.load(f)

ts.set_token(config["tushare_token"])
pro = ts.pro_api()

# 获取股票数据并保存到 CSV 文件
def fetch_and_save_data(ts_code, start_date, end_date, file_path):
    print(f"获取股票 {ts_code} 的数据，从 {start_date} 到 {end_date}...")
    data = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    data.sort_values("trade_date", inplace=True)
    data.to_csv(file_path, index=False)
    print(f"数据已保存到 {file_path}")

# 转换时间周期并重新计算均线
def resample_data(data, period, fast_window=10, slow_window=20):
    """
    按指定周期（周、月）重采样数据并重新计算均线
    """
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    data.set_index("trade_date", inplace=True)

    ohlc_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "vol": "sum"
    }
    resampled_data = data.resample(period).apply(ohlc_dict).dropna()

    # 重新计算快慢均线
    resampled_data["fast_ma"] = resampled_data["close"].rolling(fast_window).mean()
    resampled_data["slow_ma"] = resampled_data["close"].rolling(slow_window).mean()
    resampled_data.reset_index(inplace=True)
    return resampled_data

# 绘制回测结果
def plot_backtest_result(data, trades, title):
    plt.style.use("dark_background")
    plt.figure(figsize=(14, 7))

    plt.plot(data["trade_date"], data["close"], label="收盘价", alpha=0.7, color="#1f77b4")
    plt.plot(data["trade_date"], data["fast_ma"], label="快均线 (10)", linestyle="--", color="#ff7f0e", alpha=0.7)
    plt.plot(data["trade_date"], data["slow_ma"], label="慢均线 (20)", linestyle="--", color="#2ca02c", alpha=0.7)
    plt.scatter(
        [trade["date"] for trade in trades if trade["action"] == "BUY"],
        [trade["price"] for trade in trades if trade["action"] == "BUY"],
        label="买入信号",
        marker="^",
        color="#00ff00"
    )
    plt.scatter(
        [trade["date"] for trade in trades if trade["action"] == "SELL"],
        [trade["price"] for trade in trades if trade["action"] == "SELL"],
        label="卖出信号",
        marker="v",
        color="#ff4500"
    )
    plt.title(title, fontsize=14, color="white")
    plt.xlabel("日期", fontsize=12, color="white")
    plt.ylabel("价格", fontsize=12, color="white")
    plt.legend(loc="best", fontsize=10, facecolor="black")
    plt.grid(color="#555555", linestyle="--", linewidth=0.5)
    plt.xticks(rotation=45, color="white")
    plt.yticks(color="white")
    plt.tight_layout()
    plt.show()

# 主程序
if __name__ == "__main__":
    ts_code = input("请输入股票代码（如 000001.SZ）：")
    start_date = input("请输入开始日期（如 20230101）：")
    end_date = input("请输入结束日期（如 20231231）：")
    period = input("请选择时间周期（日/周/月，默认日）：").strip() or "D"

    if period.upper() not in ["D", "W", "M"]:
        print("无效的时间周期！默认使用日线数据。")
        period = "D"

    file_path = f"data/{ts_code.replace('.', '_')}.csv"
    fetch_and_save_data(ts_code, start_date, end_date, file_path)

    data = pd.read_csv(file_path)
    data = data.sort_values(by="trade_date")

    if period.upper() == "W":
        data = resample_data(data, "W")
    elif period.upper() == "M":
        data = resample_data(data, "M")

    print("\n=== 调试：重采样数据列 ===")
    print(data.head())

    print("\n=== 多策略组合回测 ===")
    ma_strategy = MaStockStrategy(data.copy())
    rsi_strategy = RsiStockStrategy(data.copy())

    # 确保计算信号
    ma_strategy.calculate_signals()
    rsi_strategy.calculate_signals()

    # 更新 data
    data = ma_strategy.data
    print("\n=== 调试：均线数据列 ===")
    print(data[["trade_date", "close", "fast_ma", "slow_ma"]].head())

    manager = StrategyManager([ma_strategy, rsi_strategy])
    final_trades = manager.execute_trades()

    print("\n交易记录：")
    for trade in final_trades:
        print(trade)

    title = f"多策略组合回测结果 - {period.upper()}线"
    plot_backtest_result(data, final_trades, title)

