class StrategyManager:
    """
    策略管理器：管理多个策略并整合交易信号
    """
    def __init__(self, strategies):
        self.strategies = strategies  # 策略列表
        self.data = strategies[0].data  # 使用第一个策略的数据作为基准
        self.final_trades = []  # 最终交易记录

    def combine_signals(self):
        """
        整合多个策略的交易信号
        """
        self.data["combined_signal"] = 0

        # 收集每个策略的信号
        for strategy in self.strategies:
            strategy.calculate_signals()
            self.data["signal"] = strategy.data["signal"]
            self.data["combined_signal"] += self.data["signal"]

        # 生成最终交易信号
        self.data["final_signal"] = 0
        self.data.loc[self.data["combined_signal"] > 0, "final_signal"] = 1  # 多数策略买入
        self.data.loc[self.data["combined_signal"] < 0, "final_signal"] = -1  # 多数策略卖出

    def execute_trades(self):
        """
        执行交易
        """
        self.combine_signals()

        position = 0  # 当前仓位状态
        for index, row in self.data.iterrows():
            signal = row["final_signal"]
            price = row["close"]
            date = row["trade_date"]

            if signal == 1 and position == 0:  # 买入
                position = 1
                self.final_trades.append({"date": date, "action": "BUY", "price": price})
                print(f"[{date}] 多策略组合信号：买入，价格：{price}")

            elif signal == -1 and position == 1:  # 卖出
                position = 0
                self.final_trades.append({"date": date, "action": "SELL", "price": price})
                print(f"[{date}] 多策略组合信号：卖出，价格：{price}")

        return self.final_trades
