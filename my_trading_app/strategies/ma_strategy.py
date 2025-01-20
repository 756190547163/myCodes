import pandas as pd

class MaStockStrategy:
    """
    双均线策略：计算快均线和慢均线信号，并生成买卖信号
    """
    def __init__(self, data, fast_window=10, slow_window=20):
        self.data = data
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.position = 0  # 当前持仓状态
        self.trades = []  # 交易记录

    def calculate_signals(self):
        """
        计算快均线、慢均线和交易信号
        """
        self.data["fast_ma"] = self.data["close"].rolling(self.fast_window).mean()
        self.data["slow_ma"] = self.data["close"].rolling(self.slow_window).mean()
        self.data["signal"] = 0
        self.data.loc[self.data["fast_ma"] > self.data["slow_ma"], "signal"] = 1  # 买入信号
        self.data.loc[self.data["fast_ma"] <= self.data["slow_ma"], "signal"] = -1  # 卖出信号
        return self.data

    def execute_trades(self):
        """
        执行买卖操作
        """
        for index, row in self.data.iterrows():
            signal = row["signal"]
            price = row["close"]
            date = row["trade_date"]

            if signal == 1 and self.position == 0:  # 买入
                self.position = 1
                self.trades.append({"date": date, "action": "BUY", "price": price})
                print(f"[{date}] 买入，价格：{price}")

            elif signal == -1 and self.position == 1:  # 卖出
                self.position = 0
                self.trades.append({"date": date, "action": "SELL", "price": price})
                print(f"[{date}] 卖出，价格：{price}")

        return self.trades

    def extract_trade_signals(self):
        """
        提取买入和卖出信号
        """
        buy_signals = self.data[(self.data["signal"] == 1) & (self.data["signal"].shift(1) != 1)]
        sell_signals = self.data[(self.data["signal"] == -1) & (self.data["signal"].shift(1) != -1)]
        return buy_signals, sell_signals
