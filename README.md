# 个人代码文档

打包命令:
1. macOS :pyinstaller --hidden-import=plyer.platforms.macos.notification --onefile --add-data "config/tushare_config.json:config" simulate_trading_extended.py
2. windows:pyinstaller --hidden-import=plyer.platforms.win.notification --onefile --add-data "config/tushare_config.json;config" simulate_trading_extended.py

## **注册与配置**

1. [注册并登录 Tushare](https://tushare.pro/register?reg=577133)
2. 登录后，点击头像进入 **用户中心** -> **接口Token** -> 复制 Token。
3. 打开配置文件 `/config/tushare_config.json`，将 `"你的Token"` 替换为复制的 Token。

---

## **脚本进化日志**

### **V1版本**
- 功能：获取历史复权数据。
- 策略：基于技术指标组合，包括 **RSI**、**布林带** 和 **MACD** 信号。

### **V2版本**
- 在 V1 的基础上新增：
  - 趋势信号。
  - 成交量分析。
  - 实时日行情数据计算。

### **V3版本**
- 在 V2 的基础上新增：
  - 涨停捕捉功能。
  - 动态止盈逻辑。
  - 止亏卖出信号。
- 特性：  
  - 更适合大盘股操作。
  - 风险承受等级较低，盈亏幅度较小。

### **V4版本**
- 在 V3 的基础上新增：
  - 新闻数据获取功能。
  - 新闻情绪分析，生成情绪分数。
  - 基础线性回归，用于预测下一交易日价格。
- 调整：
  - 止盈逻辑优化。
  - 遇到妖股时，可抓涨停，不按基本逻辑止盈。
- 特性：
  - 风险承受等级：中。
  - 盈亏幅度较高。
  - 成功案例：捕捉到 **巨轮智能** 等妖股。

---

## **未来开发计划**

### **V5版本**
- **新增功能**：在线 Web 版。
- **技术实现**：
  - 开辟 API 接口。
  - 前端采用 **CSS3**、**HTML** 和 **JavaScript**，实现炫酷交互效果。

### **V6版本**
- **新增功能**：仓位控制。
- **实现目标**：
  - 动态计算持仓股票。
  - 结合股票模型，实现合理分仓投资策略。

### **V7版本**
- **新增功能**：风险承受等级选项。
- **优化策略**：
  - 自动切换不同的策略模型。
  - 适合不同风险偏好的投资者。
  - 同时适配大小盘交易。

---

通过不断优化和迭代，这个脚本将逐步完善，成为强大的智能股票分析工具！
