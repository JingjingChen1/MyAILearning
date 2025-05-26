# config.py
"""
配置文件 - 存储所有可配置的参数

为什么需要配置文件？
1. 集中管理所有设置，方便修改
2. 避免在代码中写死参数（硬编码）
3. 不同环境可以使用不同的配置
4. 让代码更容易维护

这个文件定义了项目中使用的所有常量和配置参数
其他文件通过 from config import XXX 来使用这些配置
"""

# ==== 股票和加密货币配置 ====
# 定义我们支持分析的股票代码列表
STOCK_SYMBOLS = [
    'AAPL',    # 苹果公司
    'MSFT',    # 微软公司 
    'GOOGL',   # 谷歌公司
    'AMZN',    # 亚马逊公司
    'TSLA',    # 特斯拉公司
    'NVDA',    # 英伟达公司
    'META'     # Meta公司（原Facebook）
]

# 加密货币代码（在Yahoo Finance中的表示方式）
CRYPTO_SYMBOLS = [
    'BTC-USD',  # 比特币对美元
    'ETH-USD',  # 以太坊对美元
    'BNB-USD'   # 币安币对美元
]

# ==== API配置 ====
# API密钥配置（实际使用时需要申请真实的API密钥）
# 这里使用"demo"表示演示版本，功能有限
ALPHA_VANTAGE_API_KEY = "demo"  # Alpha Vantage新闻API的密钥
FRED_API_KEY = "demo"           # 美联储经济数据API的密钥

# API请求配置
API_TIMEOUT = 30                # API请求超时时间（秒）
MAX_RETRIES = 3                 # 请求失败时的最大重试次数

# ==== 经济指标配置 ====
# 定义要获取的经济指标及其在FRED数据库中的标识符
ECONOMIC_INDICATORS = {
    'GDP': 'GDPC1',             # GDP（国内生产总值）
    'Inflation': 'CPIAUCSL',    # CPI（消费者价格指数，通胀指标）
    'Unemployment': 'UNRATE',   # 失业率
    'Interest_Rate': 'FEDFUNDS' # 联邦基金利率
}

# ==== 机器学习模型配置 ====
# LSTM神经网络的超参数设置
LSTM_CONFIG = {
    'sequence_length': 30,      # 使用过去30天的数据来预测
    'epochs': 50,               # 训练轮数
    'batch_size': 32,           # 每批训练的样本数
    'learning_rate': 0.001,     # 学习率
    'validation_split': 0.2,    # 验证集比例
    'early_stopping_patience': 10,  # 早停耐心值
    'lr_reduction_patience': 5   # 学习率衰减耐心值
}

# 特征工程配置
FEATURE_CONFIG = {
    'ma_periods': [5, 20, 50],  # 移动平均线周期
    'rsi_period': 14,           # RSI指标周期
    'macd_fast': 12,            # MACD快线周期
    'macd_slow': 26,            # MACD慢线周期
    'macd_signal': 9,           # MACD信号线周期
    'volatility_period': 20     # 波动率计算周期
}

# ==== 投资策略配置 ====
# 风险偏好级别
RISK_LEVELS = ['保守', '平衡', '激进']

# 每种风险级别的参数设置
RISK_PARAMS = {
    '保守': {
        'threshold': 0.02,      # 需要2%以上预期收益才考虑买入
        'max_position': 0.1,    # 单只股票最大仓位10%
        'confidence_min': 0.7   # 最低信心度要求
    },
    '平衡': {
        'threshold': 0.0,       # 0%以上预期收益就考虑
        'max_position': 0.2,    # 单只股票最大仓位20%
        'confidence_min': 0.6   # 最低信心度要求
    },
    '激进': {
        'threshold': -0.05,     # 即使-5%预期收益也可能买入
        'max_position': 0.3,    # 单只股票最大仓位30%
        'confidence_min': 0.5   # 最低信心度要求
    }
}

# ==== 数据获取配置 ====
# 股票数据获取的默认参数
DATA_CONFIG = {
    'default_period': '1y',     # 默认获取1年的数据
    'min_data_points': 100,     # 最少数据点数量
    'max_news_articles': 10     # 最多分析的新闻数量
}

# ==== Streamlit界面配置 ====
# Web界面的配置参数
UI_CONFIG = {
    'page_title': '智能投资决策助手',
    'page_icon': '📈',
    'layout': 'wide',
    'sidebar_width': 300,
    'chart_height': 400,
    'default_stocks': ['AAPL', 'MSFT'],  # 默认选中的股票
    'default_risk': '平衡'               # 默认风险偏好
}

# ==== 其他常量 ====
# 时间相关常量
SECONDS_PER_DAY = 86400
TRADING_DAYS_PER_YEAR = 252

# 数值格式化
PRICE_DECIMAL_PLACES = 2
PERCENTAGE_DECIMAL_PLACES = 2
RATIO_DECIMAL_PLACES = 3

# 错误消息
ERROR_MESSAGES = {
    'no_data': '无法获取数据，请检查网络连接',
    'insufficient_data': '数据不足，无法进行分析',
    'model_training_failed': '模型训练失败',
    'prediction_failed': '价格预测失败'
}

# 成功消息
SUCCESS_MESSAGES = {
    'data_loaded': '数据加载成功',
    'model_trained': '模型训练完成',
    'analysis_complete': '分析完成'
}

# 如果这个文件被直接运行（用于测试配置是否正确）
if __name__ == "__main__":
    print("=== 配置文件测试 ===")
    print(f"支持的股票数量: {len(STOCK_SYMBOLS)}")
    print(f"股票列表: {STOCK_SYMBOLS}")
    print(f"LSTM序列长度: {LSTM_CONFIG['sequence_length']}")
    print(f"风险级别: {RISK_LEVELS}")
    print("配置文件加载正常！")