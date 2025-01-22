# app.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import pandas as pd

# 导入你封装好的交易逻辑函数
from trading_system import (
    run_backtest_and_predict,
    get_realtime_signal,
    fetch_historical_data,
    ExtendedStrategy,
    AutoTrader
)

app = Flask(__name__)
CORS(app)  # 若前端HTML不在同源域名下需要跨域

# ========== 全局存储(仅示例) ==========
# 当用户完成历史回测后，会初始化一个 trader / strategy，
# 以便后续调用实时数据时还能复用同一个账户状态、策略等
# 实际项目建议使用数据库或更专业的方式管理会话、账户
global_trader = None
global_strategy = None
global_historical_data = None
global_ts_code = None

@app.route("/")
def index():
    """返回前端页面"""
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """
    处理回测与预测
    前端通过 JSON POST 提交 {ts_code, start_date, input_year, input_money, predict_option}
    返回: { logs: [...], prediction: {...} or null, error: str or null }
    """
    data = request.json
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    ts_code = data.get("ts_code")
    start_date = data.get("start_date")
    input_year = data.get("input_year")
    input_money = data.get("input_money")
    predict_option = data.get("predict_option")

    # 调用交易系统函数
    logs, prediction = run_backtest_and_predict(ts_code, start_date, input_year, input_money, predict_option)
    if isinstance(logs, dict) and logs.get("error"):
        # 说明获取历史数据失败或别的错误
        return jsonify(logs), 400

    # 如果没出错，则把 trader/strategy/historical_data 存到全局变量
    global global_trader, global_strategy, global_historical_data, global_ts_code

    # 1) 构造一个新的 Trader / Strategy (因为 run_backtest_and_predict 内部新建了临时 trader)
    #    如果想持续使用该 trader，就需要在 run_backtest_and_predict 里把 trader返回出来
    #    这里演示：我们先重新 fetch 一遍 historical_data
    #    也可以在 run_backtest_and_predict 末尾专门返回trader和strategy
    #    但需要你改 trading_system.py 里的相应函数返回
    from trading_system import AutoTrader, ExtendedStrategy, fetch_historical_data
    # 使用当前日期作为结束日期
    end_date = pd.Timestamp.now().strftime('%Y%m%d')
    historical_data = fetch_historical_data(ts_code, start_date, end_date, input_year)
    if historical_data is not None and not historical_data.empty:
        global_trader = AutoTrader(initial_balance=float(input_money))
        global_strategy = ExtendedStrategy(historical_data.copy())
        # 再做一次 generate_signals()
        global_strategy.data = global_strategy.generate_signals()
        global_historical_data = historical_data
        global_ts_code = ts_code
    else:
        # 如果这里也失败了，就不存全局了
        pass

    response = {
        "logs": logs,             # 回测过程的日志(列表)
        "prediction": prediction, # 预测结果(或None)
        "error": None
    }
    return jsonify(response), 200


@app.route("/api/realtime", methods=["GET"])
def api_realtime():
    """
    获取最新实时数据并执行买卖判断
    必须先在 /api/analyze 成功后才有 global_trader / global_strategy
    """
    if not global_trader or not global_strategy or not global_ts_code or global_historical_data is None:
        return jsonify({"error": "请先执行 /api/analyze 进行初始化"}), 400
    from trading_system import get_realtime_signal
    ts_code = request.args.get("ts_code")
    realtime_info = get_realtime_signal(ts_code, global_historical_data, global_trader, global_strategy)
    if realtime_info is None:
        return jsonify({"error": "无法获取实时数据"}), 400

    return jsonify(realtime_info), 200


if __name__ == "__main__":
    app.run(debug=True)
