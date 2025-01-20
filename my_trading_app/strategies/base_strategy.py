class BaseStrategy:
    """
    策略基类：所有策略需要继承此类
    """
    def __init__(self, data):
        self.data = data
        self.trades = []

    def calculate_signals(self):
        """
        计算信号的方法，需子类实现
        """
        raise NotImplementedError

    def execute_trades(self):
        """
        执行买卖操作的方法，需子类实现
        """
        raise NotImplementedError
