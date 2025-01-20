import pandas as pd
from strategies.ma_strategy import MaStockStrategy

class RealtimeMaStrategy(MaStockStrategy):
    """
    实时双均线策略，继承双均线策略类
    """
    def analyze_realtime_data(self, realtime_data):
        """
        分析实时数据并生成交易信号
        """
        # 模拟将实时数据添加到已有数据中
        self.data = pd.concat([self.data, realtime_data], ignore_index=True)
        self.data = self.calculate_signals()

        # 输出当前数据的快慢均线
        print(f"最新数据：\n{self.data[['trade_date', 'close', 'fast_ma', 'slow_ma']].tail()}")

        # 生成信号
        last_signal = self.data.iloc[-1]["signal"]
        if last_signal == 1:
            print(f"实时信号：买入 ({realtime_data['trade_date'].iloc[-1]})，价格：{realtime_data['close'].iloc[-1]}")
        elif last_signal == -1:
            print(f"实时信号：卖出 ({realtime_data['trade_date'].iloc[-1]})，价格：{realtime_data['close'].iloc[-1]}")
        else:
            print("无交易信号。")
        return last_signal
