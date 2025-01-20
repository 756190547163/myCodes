class RsiStockStrategy:
    """
    RSI 策略：根据相对强弱指标生成买卖信号
    """
    def __init__(self, data, rsi_period=14, overbought=70, oversold=30):
        self.data = data
        self.rsi_period = rsi_period
        self.overbought = overbought
        self.oversold = oversold
        self.trades = []

    def calculate_rsi(self):
        """
        计算 RSI 指标
        """
        delta = self.data["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(self.rsi_period).mean()
        avg_loss = loss.rolling(self.rsi_period).mean()

        rs = avg_gain / avg_loss
        self.data["rsi"] = 100 - (100 / (1 + rs))

    def calculate_signals(self):
        """
        计算 RSI 信号
        """
        self.calculate_rsi()
        self.data["signal"] = 0
        self.data.loc[self.data["rsi"] < self.oversold, "signal"] = 1  # 买入信号
        self.data.loc[self.data["rsi"] > self.overbought, "signal"] = -1  # 卖出信号

    def execute_trades(self):
        """
        基于 RSI 信号执行买卖操作
        """
        self.calculate_signals()

        for index, row in self.data.iterrows():
            rsi = row["rsi"]
            price = row["close"]
            date = row["trade_date"]

            if rsi < self.oversold:  # 超卖，买入
                self.trades.append({"date": date, "action": "BUY", "price": price})
                print(f"[{date}] RSI={rsi:.2f}，超卖买入，价格：{price}")
            elif rsi > self.overbought:  # 超买，卖出
                self.trades.append({"date": date, "action": "SELL", "price": price})
                print(f"[{date}] RSI={rsi:.2f}，超买卖出，价格：{price}")

        return self.trades
