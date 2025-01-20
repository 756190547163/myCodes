import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
import requests
import numpy as np

class ExtendedStrategy:
    """
    扩展策略，支持 RSI、MACD 和布林带，并动态优化参数，同时结合情绪分析、趋势检测、成交量和价格预测
    """
    def __init__(self, data, rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9, bb_period=20, sentiment_api_key=None, is_realtime=False):
        self.data = data
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bb_period = bb_period
        self.model = LinearRegression()
        self.sentiment_api_key = sentiment_api_key
        self.is_realtime = is_realtime  # 区分历史数据与实时数据
        self.dynamic_stop_profit = 1.1  # 动态止盈初始值
        self._rate_limit_logged = False  # 避免重复打印日志

    def calculate_rsi(self):
        """
        计算 RSI 指标
        """
        delta = self.data["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()

        rs = avg_gain / avg_loss
        self.data["rsi"] = 100 - (100 / (1 + rs))

    def calculate_macd(self):
        """
        计算 MACD 指标
        """
        ema_fast = self.data["close"].ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = self.data["close"].ewm(span=self.macd_slow, adjust=False).mean()
        self.data["macd"] = ema_fast - ema_slow
        self.data["macd_signal"] = self.data["macd"].ewm(span=self.macd_signal, adjust=False).mean()
        self.data["macd_hist"] = self.data["macd"] - self.data["macd_signal"]

    def calculate_bollinger_bands(self):
        """
        计算布林带
        """
        self.data["bb_middle"] = self.data["close"].rolling(window=self.bb_period).mean()
        self.data["bb_std"] = self.data["close"].rolling(window=self.bb_period).std()
        self.data["bb_upper"] = self.data["bb_middle"] + (2 * self.data["bb_std"])
        self.data["bb_lower"] = self.data["bb_middle"] - (2 * self.data["bb_std"])

    def calculate_volume_signals(self):
        """
        根据成交量生成信号
        """
        self.data["vol_ma5"] = self.data["vol"].rolling(window=5).mean()
        self.data["vol_ma10"] = self.data["vol"].rolling(window=10).mean()

        # 成交量放大信号
        self.data["vol_spike"] = (self.data["vol"] > 1.5 * self.data["vol_ma5"]).astype(int)

        # 成交量突破信号
        self.data["vol_break"] = (self.data["vol"] > self.data["vol_ma10"] * 2).astype(int)

    def detect_limit_up(self):
        """
        检测涨停信号
        """
        self.data["limit_up"] = (self.data["high"] >= self.data["close"] * 1.09).astype(int)

    def analyze_trend(self):
        """
        分析市场趋势
        """
        self.data["trend"] = self.data["close"].rolling(window=5).mean().diff()
        self.data["trend_signal"] = (self.data["trend"] > 0).astype(int) - (self.data["trend"] < 0).astype(int)

    def get_sentiment_scores(self):
        """
        获取当前时间的新闻情绪分数
        """
        from datetime import datetime
        current_date = datetime.now().strftime('%Y-%m-%d')
        sentiment_scores = [self.fetch_news_sentiment(current_date)]
        return sentiment_scores

    def fetch_news_sentiment(self, date):
        """
        API 获取情绪分数。
        """
        try:
            if not self.sentiment_api_key:
                return 0
            url = f"https://newsapi.org/v2/everything?q=stock&from={date}&sortBy=popularity&apiKey={self.sentiment_api_key}"
            response = requests.get(url)
            response.raise_for_status()
            articles = response.json().get("articles", [])
            sentiments = []
            for article in articles:
                title = article.get("title", "")
                description = article.get("description", "")
                sentiment = TextBlob(title + " " + description).sentiment.polarity
                sentiments.append(sentiment)
            return np.mean(sentiments) if sentiments else 0
        except requests.exceptions.HTTPError as http_err:
            if response.status_code == 429:
                if not self._rate_limit_logged:
                    print("新闻情绪分析跳过：超出 API 请求限制")
                    self._rate_limit_logged = True
            else:
                print(f"情绪分析 HTTP 错误: {http_err}")
            return 0
        except Exception as e:
            print(f"情绪分析错误: {e}")
            return 0

    def analyze_sentiment(self):
        """
        使用 TextBlob 分析新闻情绪并生成情绪分数
        """
        if self.is_realtime and "trade_date" in self.data.columns:
            # 获取当前情绪分数
            current_sentiment = self.get_sentiment_scores()[0]  # 返回单个值
            # 将当前情绪分数广播到整个数据框
            self.data["sentiment"] = current_sentiment
        else:
            self.data["sentiment"] = 0

    def optimize_parameters(self):
        """
        动态优化参数以适应市场波动
        """
        # 根据市场波动动态调整 RSI 周期
        recent_volatility = self.data["close"].pct_change().rolling(window=10).std().iloc[-1]
        if recent_volatility > 0.02:
            self.rsi_period = 10  # 提高 RSI 灵敏度
        else:
            self.rsi_period = 14

        # 动态调整布林带周期
        if recent_volatility > 0.03:
            self.bb_period = 15
        else:
            self.bb_period = 20

    def predict_next_day(self):
        """
        使用线性回归预测下一交易日价格
        """
        self.data.dropna(inplace=True)
        features = ["open", "high", "low", "close", "vol"]
        target = "close"
        X = self.data[features].values[:-1]
        y = self.data[target].values[1:]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        self.model.fit(X_scaled, y)
        last_day_features = scaler.transform(self.data[features].iloc[-1].values.reshape(1, -1))
        predicted_price = self.model.predict(last_day_features)[0]

        return predicted_price

    def generate_signals(self):
        """
        生成综合信号
        """
        self.optimize_parameters()
        self.calculate_rsi()
        self.calculate_macd()
        self.calculate_bollinger_bands()
        self.calculate_volume_signals()
        self.analyze_sentiment()
        self.detect_limit_up()
        self.analyze_trend()

        # 初始化信号列
        self.data["signal"] = 0

        # RSI 信号
        self.data.loc[self.data["rsi"] < 30, "signal"] += 1  # 超卖买入
        self.data.loc[self.data["rsi"] > 70, "signal"] -= 1  # 超买卖出

        # MACD 信号
        self.data.loc[self.data["macd"] > self.data["macd_signal"], "signal"] += 1  # 多头
        self.data.loc[self.data["macd"] < self.data["macd_signal"], "signal"] -= 1  # 空头

        # 布林带信号
        self.data.loc[self.data["close"] < self.data["bb_lower"], "signal"] += 1  # 跌破下轨买入
        self.data.loc[self.data["close"] > self.data["bb_upper"], "signal"] -= 1  # 突破上轨卖出

        # 涨停信号
        self.data.loc[self.data["limit_up"] == 1, "signal"] += 2  # 强制买入

        # 趋势信号
        self.data.loc[self.data["trend_signal"] > 0, "signal"] += 1
        self.data.loc[self.data["trend_signal"] < 0, "signal"] -= 1

        # 成交量信号
        self.data.loc[self.data["vol_spike"] == 1, "signal"] += 1  # 成交量放大买入
        self.data.loc[self.data["vol_break"] == 1, "signal"] += 2  # 成交量突破强制买入

        # 情绪信号
        self.data.loc[self.data["sentiment"] > 0.5, "signal"] += 1
        self.data.loc[self.data["sentiment"] < -0.5, "signal"] -= 1

        # 动态止盈逻辑
        self.data["dynamic_stop_profit"] = self.dynamic_stop_profit
        self.data.loc[self.data["close"] > self.data["close"].rolling(window=5).max(), "dynamic_stop_profit"] = 0.95
        self.data["stop_profit"] = (self.data["close"] >= self.data["dynamic_stop_profit"] * self.data["close"].rolling(window=5).mean()).astype(int)
        self.data.loc[self.data["stop_profit"] == 1, "signal"] -= 2  # 动态止盈卖出

        # 添加止损卖出信号
        self.data["stop_loss"] = (self.data["close"] <= self.data["close"].rolling(window=5).mean() * 0.9).astype(int)
        self.data.loc[self.data["stop_loss"] == 1, "signal"] -= 2  # 止损卖出

        return self.data
