# 🤖 智能投资决策助手

基于AI技术的股票投资分析系统，整合技术分析、情感分析、深度学习预测和智能策略生成。

## 📋 功能特点

### 🔍 多维度分析
- **技术分析**: 移动平均线、RSI、MACD、布林带等经典指标
- **情感分析**: 基于新闻的市场情感识别和量化
- **AI预测**: LSTM神经网络进行价格趋势预测
- **策略生成**: 个性化投资建议和仓位管理

### 🎯 智能特性
- **风险匹配**: 根据投资者风险偏好定制建议
- **组合优化**: 智能投资组合配置和资金分配
- **实时分析**: 基于最新市场数据的动态分析
- **可视化**: 交互式图表展示分析结果

## 🚀 快速开始

### 1. 环境要求
- Python 3.8 或更高版本
- 8GB+ RAM (推荐)
- 稳定的网络连接

### 2. 安装和运行

```bash

# 2. 进入项目目录
cd investment_analyzer

# 3. 运行程序（会自动安装依赖）
streamlit run main.py

3. 使用步骤
在浏览器中打开显示的网址 (通常是 http://localhost:8501)
在左侧面板选择要分析的股票
设置风险偏好和分析参数
点击"开始智能分析"
等待分析完成，查看详细报告



🔧 技术架构
数据层
Yahoo Finance API: 获取股票价格、交易量等市场数据
Alpha Vantage API: 获取新闻和情感数据
FRED API: 获取宏观经济指标
分析层
技术指标计算: Pandas数据处理和NumPy数学计算
情感分析: Transformers库的FinBERT模型
价格预测: TensorFlow/Keras的LSTM神经网络
策略优化: Scikit-learn机器学习算法
界面层
Streamlit: Web界面框架
Plotly: 交互式图表可视化
响应式设计: 适配不同设备屏幕
📊 分析流程
1. 数据收集 (30%)

# 获取股票历史数据
stock_data = collector.fetch_stock_data('AAPL', '1y')

# 计算技术指标
indicators = calculate_technical_indicators(stock_data)

# 获取新闻情感数据
sentiment = collector.fetch_news_sentiment('AAPL')
2. 特征工程 (20%)

# 整合多维特征
features = prepare_features(
    stock_data=stock_data,
    sentiment_data=sentiment,
    economic_data=economic_indicators
)
3. AI预测 (30%)

# 训练LSTM模型
model = predictor.train_model(features)

# 预测未来价格
prediction = predictor.predict_next_price(recent_data)
4. 策略生成 (20%)

# 生成投资建议
strategy = generator.generate_strategy(
    predictions, sentiment, market_data
)

# 优化投资组合
portfolio = generator.generate_portfolio_advice(
    strategies, risk_preference, capital
)



📈 性能优化
提升分析速度
快速模式: 减少AI训练轮数
缓存机制: 重用已分析的数据
并行处理: 多股票同时分析
提高预测精度
特征工程: 增加更多有效特征
模型调优: 优化神经网络参数
集成学习: 结合多种预测方法
