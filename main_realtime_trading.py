import tushare as ts
import pandas as pd
import time
import json
from strategies.realtime_ma_strategy import RealtimeMaStrategy

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
        else:
            print("无交易操作。")

    def get_account_status(self):
        """
        输出当前账户状态
        """
        total_assets = self.balance + self.position * self.hold_price
        profit_loss = total_assets - self.initial_balance
        print(f"账户余额：{self.balance:.2f}，持仓股数：{self.position}，当前持仓成本：{self.hold_price:.2f}")
        print(f"总资产：{total_assets:.2f}，收益/亏损：{profit_loss:.2f}")

if __name__ == "__main__":
    ts_code = input("请输入股票代码（如 000001.SZ）：")
    start_date = input("请输入开始日期（如 20220101）：")
    end_date = input("请输入结束日期（如 20231231）：")

    # 获取历史数据
    historical_data = fetch_historical_data(ts_code, start_date, end_date)
    if historical_data is None or historical_data.empty:
        print("无法获取历史数据，程序退出。")
        exit()

    # 初始化策略和交易器
    strategy = RealtimeMaStrategy(historical_data.copy(), fast_window=5, slow_window=10)
    trader = AutoTrader(initial_balance=100000)

    # 模拟时间流逝
    for index, row in historical_data.iterrows():
        print(f"\n日期：{row['trade_date']}")
        # 获取当天数据
        daily_data = pd.DataFrame([row])

        # 分析信号
        signal = strategy.analyze_realtime_data(daily_data)

        # 执行交易
        trader.execute_trade(signal, row["close"])

        # 输出账户状态
        trader.get_account_status()

        # 模拟一天时间流逝
        time.sleep(5)
