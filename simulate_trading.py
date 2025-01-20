import tushare as ts
import pandas as pd
import time
import json
import matplotlib.pyplot as plt

from matplotlib import rcParams


# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

# 加载 Tushare 配置
with open("config/tushare_config.json") as f:
    config = json.load(f)

ts.set_token(config["tushare_token"])
pro = ts.pro_api()

# 获取历史数据
def fetch_historical_data(ts_code, start_date, end_date):
    """
    获取股票历史数据
    """
    try:
        data = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if data.empty:
            print(f"无法获取 {ts_code} 的历史数据。")
            return None
        data = data[["trade_date", "open", "high", "low", "close", "vol"]]
        data.sort_values("trade_date", inplace=True)
        return data
    except Exception as e:
        print(f"获取历史数据失败：{e}")
        return None

# 计算技术指标
def calculate_indicators(data):
    """
    计算 RSI、MACD 和布林带指标
    """
    # RSI
    delta = data["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    data["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = data["close"].ewm(span=12, adjust=False).mean()
    ema_26 = data["close"].ewm(span=26, adjust=False).mean()
    data["macd"] = ema_12 - ema_26
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    rolling_mean = data["close"].rolling(window=20).mean()
    rolling_std = data["close"].rolling(window=20).std()
    data["bollinger_upper"] = rolling_mean + (rolling_std * 2)
    data["bollinger_lower"] = rolling_mean - (rolling_std * 2)

    return data

# 生成多策略信号
def generate_combined_signal(data):
    """
    根据 RSI、MACD 和布林带信号生成综合交易信号
    """
    data["rsi_signal"] = 0
    data.loc[data["rsi"] < 30, "rsi_signal"] = 1  # RSI 超卖买入信号
    data.loc[data["rsi"] > 70, "rsi_signal"] = -1  # RSI 超买卖出信号

    data["macd_signal"] = 0
    data.loc[(data["macd"] > data["macd_signal"]) &
             (data["macd"].shift(1) <= data["macd_signal"].shift(1)), "macd_signal"] = 1  # MACD 金叉买入
    data.loc[(data["macd"] < data["macd_signal"]) &
             (data["macd"].shift(1) >= data["macd_signal"].shift(1)), "macd_signal"] = -1  # MACD 死叉卖出

    data["bollinger_signal"] = 0
    data.loc[data["close"] < data["bollinger_lower"], "bollinger_signal"] = 1  # 价格突破下轨买入信号
    data.loc[data["close"] > data["bollinger_upper"], "bollinger_signal"] = -1  # 价格突破上轨卖出信号

    # 综合信号
    data["combined_signal"] = data[["rsi_signal", "macd_signal", "bollinger_signal"]].sum(axis=1)
    data["combined_signal"] = data["combined_signal"].apply(lambda x: 1 if x > 1 else (-1 if x < -1 else 0))

    return data

# 模拟交易逻辑
class AutoTrader:
    def __init__(self, initial_balance=100000):
        self.position = 0  # 当前持仓股数
        self.balance = initial_balance  # 账户余额
        self.initial_balance = initial_balance  # 初始本金
        self.hold_price = 0  # 持仓成本价

    def execute_trade(self, signal, price):
        """
        根据信号执行交易
        """
        if signal == 1 and self.position == 0:  # 买入
            self.position = self.balance // price  # 计算可买入股数
            self.hold_price = price  # 更新持仓成本价
            self.balance -= self.position * price  # 扣除账户余额
            print(f"执行买入操作，价格：{price:.2f}，买入股数：{self.position}，剩余余额：{self.balance:.2f}")
        elif signal == -1 and self.position > 0:  # 卖出
            profit = self.position * (price - self.hold_price)  # 计算收益
            self.balance += self.position * price  # 更新账户余额
            print(f"执行卖出操作，价格：{price:.2f}，卖出股数：{self.position}，收益：{profit:.2f}")
            self.position = 0  # 清空仓位
            self.hold_price = 0  # 清空成本价

    def get_account_status(self, current_price=None):
        """
        输出并返回当前账户状态，包括浮盈浮亏
        """
        unrealized_pnl = self.calculate_unrealized_pnl(current_price) if current_price else 0.0
        # 如果 current_price 为 None，就用 hold_price 计算总资产，否则用 current_price
        price_for_valuation = current_price if current_price else self.hold_price
        total_assets = self.balance + self.position * price_for_valuation
        profit_loss = total_assets - self.initial_balance

        # 打印调试信息
        print(f"账户余额：{self.balance:.2f}，持仓股数：{self.position}，当前持仓成本：{self.hold_price:.2f}")
        print(f"浮盈/浮亏：{unrealized_pnl:.2f}，总资产：{total_assets:.2f}，收益/亏损：{profit_loss:.2f}")

        # 返回字典，以便在其他模块 / 前端看到
        return {
            "balance": round(self.balance, 2),
            "position": int(self.position),
            "hold_price": round(self.hold_price, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "total_assets": round(total_assets, 2),
            "profit_loss": round(profit_loss, 2)
        }

    def calculate_unrealized_pnl(self, current_price):
        """
        计算浮盈浮亏
        """
        if self.position > 0:
            unrealized_pnl = (current_price - self.hold_price) * self.position
            return unrealized_pnl
        return 0.0

# 可视化账户净值曲线
def plot_equity_curve(equity_curve):
    """
    绘制账户净值曲线
    """
    plt.figure(figsize=(14, 7))
    plt.plot(equity_curve["date"], equity_curve["equity"], label="账户净值", color="blue", linewidth=2)
    plt.title("账户净值曲线", fontsize=14)
    plt.xlabel("日期", fontsize=12)
    plt.ylabel("净值", fontsize=12)
    plt.legend(loc="best", fontsize=12)
    plt.grid()
    plt.show()

# 可视化交易信号
def plot_signals(data):
    """
    绘制交易信号和布林带
    """
    plt.figure(figsize=(14, 7))
    plt.plot(data["trade_date"], data["close"], label="收盘价", color="blue")
    plt.plot(data["trade_date"], data["bollinger_upper"], label="布林带上轨", linestyle="--", color="red")
    plt.plot(data["trade_date"], data["bollinger_lower"], label="布林带下轨", linestyle="--", color="green")

    buy_signals = data[data["combined_signal"] == 1]
    sell_signals = data[data["combined_signal"] == -1]
    plt.scatter(buy_signals["trade_date"], buy_signals["close"], label="买入信号", marker="^", color="green", s=100)
    plt.scatter(sell_signals["trade_date"], sell_signals["close"], label="卖出信号", marker="v", color="red", s=100)

    plt.title("交易信号和布林带", fontsize=14)
    plt.xlabel("日期", fontsize=12)
    plt.ylabel("价格", fontsize=12)
    plt.legend(loc="best", fontsize=12)
    plt.grid()
    plt.show()

# 主程序
if __name__ == "__main__":
    ts_code = input("请输入股票代码（如 000001.SZ）：")
    start_date = input("请输入开始日期（如 20220101）：")
    end_date = input("请输入结束日期（如 20221231）：")

    # 获取历史数据
    historical_data = fetch_historical_data(ts_code, start_date, end_date)
    if historical_data is None or historical_data.empty:
        print("无法获取历史数据，程序退出。")
        exit()

    # 计算技术指标
    historical_data = calculate_indicators(historical_data)

    # 生成综合信号
    historical_data = generate_combined_signal(historical_data)

    # 模拟交易
    trader = AutoTrader(initial_balance=100000)
    equity_curve = []
    for index, row in historical_data.iterrows():
        signal = row["combined_signal"]
        trader.execute_trade(signal, row["close"])
        equity_curve.append({
            "date": row["trade_date"],
            "equity": trader.balance + trader.position * row["close"]
        })

    equity_curve = pd.DataFrame(equity_curve)

    # 可视化结果
    plot_equity_curve(equity_curve)
    plot_signals(historical_data)
