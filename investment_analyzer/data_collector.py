# data_collector.py
"""
数据收集模块

这个模块的作用：
1. 从Yahoo Finance获取股票价格、交易量等数据
2. 从Alpha Vantage获取新闻情感数据
3. 从FRED获取经济指标数据
4. 计算技术指标（移动平均线、RSI、MACD等）

技术指标是什么？
- 基于股价、交易量计算出的数学指标
- 帮助分析股票的趋势和买卖信号
- 是技术分析的重要工具
"""

# 导入需要的库
import yfinance as yf          # Yahoo Finance数据库接口
import pandas as pd            # 数据处理库，类似于Excel
import numpy as np             # 数学计算库，处理数字和数组
import requests               # HTTP请求库，用于获取网页数据
from datetime import datetime, timedelta  # 处理日期和时间
import time                   # 时间相关功能
import warnings              # 警告管理

# 导入我们自己的配置
from config import (
    STOCK_SYMBOLS,           # 支持的股票列表
    CRYPTO_SYMBOLS,          # 支持的加密货币列表
    ECONOMIC_INDICATORS,     # 经济指标配置
    ALPHA_VANTAGE_API_KEY,   # 新闻API密钥
    FRED_API_KEY,            # 经济数据API密钥
    FEATURE_CONFIG,          # 技术指标配置
    DATA_CONFIG,             # 数据获取配置
    API_TIMEOUT,             # API超时设置
    MAX_RETRIES             # 最大重试次数
)

# 忽略一些不重要的警告信息，让输出更清晰
warnings.filterwarnings('ignore')

class MarketDataCollector:
    """
    市场数据收集器类
    
    这个类封装了所有数据获取的功能：
    - 股票价格数据
    - 新闻情感数据  
    - 经济指标数据
    - 技术指标计算
    """
    
    def __init__(self):
        """
        初始化数据收集器
        
        __init__方法是类的构造函数，创建对象时自动调用
        self参数代表类的实例本身
        """
        # 将配置文件中的参数赋值给实例属性
        self.stock_symbols = STOCK_SYMBOLS        # 可分析的股票列表
        self.crypto_symbols = CRYPTO_SYMBOLS      # 可分析的加密货币列表
        self.api_timeout = API_TIMEOUT            # API请求超时时间
        self.max_retries = MAX_RETRIES            # 最大重试次数
        
        # 打印初始化信息，让用户知道系统状态
        print(f"数据收集器初始化完成，支持 {len(self.stock_symbols)} 只股票")
    
    def fetch_stock_data(self, symbol, period=None):
        """
        获取单只股票的历史数据
        
        参数说明：
        symbol (str): 股票代码，如 'AAPL'
        period (str): 数据时间范围，如 '1y' 表示1年
        
        返回值：
        dict: 包含股票数据的字典，如果失败返回None
        
        字典结构：
        {
            'price_data': DataFrame,     # 价格数据（包含技术指标）
            'company_info': dict,        # 公司基本信息
            'symbol': str               # 股票代码
        }
        """
        
        # 如果没有指定时间范围，使用配置文件中的默认值
        if period is None:
            period = DATA_CONFIG['default_period']
        
        # 使用try-except处理可能的错误
        # try块：尝试执行的代码
        # except块：如果出错要执行的代码
        try:
            print(f"正在获取 {symbol} 的股票数据...")
            
            # 创建yfinance的Ticker对象
            # Ticker是yfinance库中的类，用于获取特定股票的信息
            ticker = yf.Ticker(symbol)
            
            # 获取历史价格数据
            # history()方法返回DataFrame，包含：
            # - Open: 开盘价
            # - High: 最高价  
            # - Low: 最低价
            # - Close: 收盘价
            # - Volume: 交易量
            data = ticker.history(period=period)
            
            # 检查是否获取到足够的数据
            if len(data) < DATA_CONFIG['min_data_points']:
                print(f"警告：{symbol} 数据点不足，只有 {len(data)} 个数据点")
                return None
            
            # 获取公司基本信息（市值、行业、员工数等）
            # info属性包含公司的详细信息
            try:
                info = ticker.info
            except Exception as e:
                print(f"获取 {symbol} 公司信息失败: {e}")
                info = {}  # 如果获取失败，使用空字典
            
            # 计算技术指标
            # 这会在原始数据上添加新的列（MA5, MA20, RSI等）
            data = self._calculate_technical_indicators(data)
            
            print(f"✅ {symbol} 数据获取成功，共 {len(data)} 个数据点")
            
            # 返回整理好的数据字典
            return {
                'price_data': data,        # 价格数据（已包含技术指标）
                'company_info': info,      # 公司信息
                'symbol': symbol           # 股票代码
            }
            
        except Exception as e:
            # 如果获取数据过程中出现任何错误，打印错误信息并返回None
            print(f"❌ 获取 {symbol} 数据失败: {e}")
            return None
    
    def _calculate_technical_indicators(self, df):
        """
        计算技术指标
        
        技术指标是基于历史价格和交易量计算的数学指标
        用于分析股票趋势和寻找买卖信号
        
        参数：
        df (DataFrame): 包含股票价格数据的DataFrame
        
        返回：
        DataFrame: 添加了技术指标列的DataFrame
        
        注意：方法名前的下划线 _ 表示这是私有方法
        私有方法通常只在类内部使用，不建议外部直接调用
        """
        
        print("正在计算技术指标...")
        
        # 1. 计算移动平均线（Moving Average, MA）
        # 移动平均线是什么？
        # - 把过去N天的收盘价加起来除以N，得到平均价格
        # - 用来平滑价格波动，判断趋势方向
        # - 短期MA向上穿越长期MA通常表示上涨趋势
        
        # 从配置文件获取移动平均线的周期设置
        ma_periods = FEATURE_CONFIG['ma_periods']  # [5, 20, 50]
        
        for period in ma_periods:
            # rolling(window=period) 创建滑动窗口
            # mean() 计算窗口内的平均值
            df[f'MA{period}'] = df['Close'].rolling(window=period).mean()
            
        # 解释rolling的工作原理：
        # 假设有价格 [10, 11, 12, 13, 14]，计算MA3：
        # 第1-2天：数据不足，返回NaN
        # 第3天：(10+11+12)/3 = 11
        # 第4天：(11+12+13)/3 = 12  
        # 第5天：(12+13+14)/3 = 13
        
        # 2. 计算RSI（相对强弱指标）
        # RSI是什么？
        # - 衡量股票买卖力量强弱的指标，范围0-100
        # - RSI > 70 通常表示超买（可能下跌）
        # - RSI < 30 通常表示超卖（可能上涨）
        
        rsi_period = FEATURE_CONFIG['rsi_period']  # 14天
        
        # 计算每日价格变化
        delta = df['Close'].diff()  # diff()计算相邻两个值的差
        
        # 分离上涨和下跌的变化
        # where()函数：如果条件为真保留原值，否则设为指定值
        gain = delta.where(delta > 0, 0)  # 只保留正数（上涨），负数设为0
        loss = -delta.where(delta < 0, 0)  # 只保留负数的绝对值（下跌）
        
        # 计算平均上涨和下跌幅度
        avg_gain = gain.rolling(window=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period).mean()
        
        # 计算相对强度
        rs = avg_gain / avg_loss
        
        # 计算RSI值
        # RSI公式：100 - (100 / (1 + RS))
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 3. 计算MACD（移动平均收敛发散指标）
        # MACD是什么？
        # - 通过两条不同周期的指数移动平均线的差值来判断趋势
        # - MACD线向上表示上涨动能增强
        # - MACD线穿越信号线通常是买卖信号
        
        # 从配置获取MACD参数
        fast_period = FEATURE_CONFIG['macd_fast']    # 12天
        slow_period = FEATURE_CONFIG['macd_slow']    # 26天
        signal_period = FEATURE_CONFIG['macd_signal'] # 9天
        
        # 计算指数移动平均线（EMA）
        # ewm()是指数加权移动平均，近期数据权重更大
        ema_fast = df['Close'].ewm(span=fast_period).mean()   # 12日EMA
        ema_slow = df['Close'].ewm(span=slow_period).mean()   # 26日EMA
        
        # MACD线 = 快线 - 慢线
        df['MACD'] = ema_fast - ema_slow
        
        # 信号线 = MACD的9日EMA
        df['MACD_Signal'] = df['MACD'].ewm(span=signal_period).mean()
        
        # MACD柱状图 = MACD线 - 信号线
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # 4. 计算波动率（Volatility）
        # 波动率是什么？
        # - 衡量价格变化剧烈程度的指标
        # - 标准差越大，价格波动越剧烈，风险越高
        # - 投资者用它来评估风险水平
        
        volatility_period = FEATURE_CONFIG['volatility_period']  # 20天
        
        # 计算收盘价的标准差作为波动率
        # std()计算标准差
        df['Volatility'] = df['Close'].rolling(window=volatility_period).std()
        
        # 5. 计算价格变化率
        # pct_change()计算百分比变化率
        # 公式：(今日价格 - 昨日价格) / 昨日价格
        df['Price_Change'] = df['Close'].pct_change()
        
        # 6. 计算交易量移动平均线
        # 交易量也是重要的技术指标
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        
        # 7. 计算布林带（Bollinger Bands）
        # 布林带用于判断价格是否偏离正常范围
        ma20 = df['MA20']  # 20日移动平均线作为中轨
        std20 = df['Close'].rolling(window=20).std()  # 20日标准差
        
        df['BB_Upper'] = ma20 + (std20 * 2)  # 上轨 = 中轨 + 2倍标准差
        df['BB_Lower'] = ma20 - (std20 * 2)  # 下轨 = 中轨 - 2倍标准差
        
        print("✅ 技术指标计算完成")
        return df
    
    def fetch_news_sentiment(self, symbol):
        """
        获取新闻情感数据
        
        什么是情感分析？
        - 使用AI技术分析新闻标题或内容的情感倾向
        - 判断新闻是积极的（positive）、消极的（negative）还是中性的（neutral）
        - 新闻情感会影响投资者心理，进而影响股价
        
        参数：
        symbol (str): 股票代码
        
        返回：
        list: 包含新闻情感分析结果的列表
        每个元素是一个字典，包含标题、情感得分、相关性等信息
        """
        
        print(f"正在获取 {symbol} 的新闻情感数据...")
        
        # 构建Alpha Vantage新闻API的URL
        # f-string语法：f"字符串{变量}"，方便插入变量值
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
        
        # 尝试多次请求，提高成功率
        for attempt in range(self.max_retries):
            try:
                print(f"  尝试第 {attempt + 1} 次请求...")
                
                # 发送HTTP GET请求获取数据
                # timeout参数设置超时时间，避免无限等待
                response = requests.get(url, timeout=self.api_timeout)
                
                # 检查HTTP状态码
                # 200表示请求成功
                if response.status_code != 200:
                    print(f"  HTTP请求失败，状态码: {response.status_code}")
                    continue
                
                # 将返回的JSON数据转换为Python字典
                # JSON是一种数据格式，类似于Python的字典
                news_data = response.json()
                
                # 检查API返回的数据结构是否正确
                if 'feed' not in news_data:
                    print(f"  API返回数据格式错误: {news_data}")
                    continue
                
                # 处理新闻数据
                sentiments = []  # 创建空列表存储情感分析结果
                
                # 遍历新闻列表，最多处理指定数量的新闻
                max_articles = DATA_CONFIG['max_news_articles']
                for article in news_data['feed'][:max_articles]:
                    
                    # 提取情感得分
                    # get()方法：如果键存在返回对应值，否则返回默认值
                    sentiment_score = float(article.get('overall_sentiment_score', 0))
                    
                    # 将每条新闻的信息整理成字典
                    sentiments.append({
                        'title': article.get('title', ''),                    # 新闻标题
                        'sentiment': sentiment_score,                         # 情感得分(-1到1)
                        'relevance': float(article.get('relevance_score', 0)), # 相关性得分
                        'time': article.get('time_published', ''),           # 发布时间
                        'summary': article.get('summary', '')[:200]          # 新闻摘要(截取前200字符)
                    })
                
                print(f"✅ 成功获取 {len(sentiments)} 条新闻")
                return sentiments
                
            except requests.exceptions.Timeout:
                print(f"  请求超时，重试中...")
                time.sleep(2)  # 等待2秒后重试
                
            except requests.exceptions.RequestException as e:
                print(f"  网络请求错误: {e}")
                time.sleep(2)
                
            except ValueError as e:
                print(f"  JSON解析错误: {e}")
                break  # JSON错误通常不是临时性的，直接跳出循环
                
            except Exception as e:
                print(f"  未知错误: {e}")
                break
        
        # 如果所有尝试都失败，返回空列表
        print(f"❌ 获取 {symbol} 新闻情感数据失败")
        return []
    
    def fetch_economic_indicators(self):
        """
        获取经济指标数据
        
        经济指标是什么？
        - 反映国家经济状况的统计数据
        - 如GDP（国内生产总值）、通胀率、失业率、利率等
        - 这些指标会影响整个股市的走势
        - 宏观经济好转通常对股市有利
        
        返回：
        dict: 包含各种经济指标数据的字典
        键是指标名称，值是包含时间序列数据的列表
        """
        
        print("正在获取经济指标数据...")
        
        economic_data = {}  # 创建空字典存储结果
        
        # 遍历配置文件中定义的每个经济指标
        for name, series_id in ECONOMIC_INDICATORS.items():
            print(f"  获取 {name} 数据...")
            
            # 尝试多次请求
            for attempt in range(self.max_retries):
                try:
                    # 构建FRED API的URL
                    # FRED是美联储经济数据库，提供免费的经济数据API
                    url = (f"https://api.stlouisfed.org/fred/series/observations?"
                          f"series_id={series_id}&"
                          f"api_key={FRED_API_KEY}&"
                          f"file_type=json&"
                          f"limit=10")  # 获取最近10个数据点
                    
                    # 发送请求
                    response = requests.get(url, timeout=self.api_timeout)
                    
                    if response.status_code != 200:
                        print(f"    HTTP错误: {response.status_code}")
                        continue
                    
                    # 解析JSON数据
                    data = response.json()
                    
                    # 检查数据格式
                    if 'observations' not in data:
                        print(f"    API数据格式错误")
                        continue
                    
                    # 处理观测数据
                    observations = data['observations']
                    if not observations:
                        print(f"    {name} 暂无数据")
                        economic_data[name] = []
                        break
                    
                    # 取最近的数据点
                    latest_data = observations[-5:]  # 最近5个数据点
                    
                    # 整理数据格式
                    processed_data = []
                    for obs in latest_data:
                        # 处理数值，'.'表示无数据
                        value = obs.get('value', '.')
                        if value != '.':
                            try:
                                value = float(value)
                            except ValueError:
                                value = None
                        else:
                            value = None
                        
                        processed_data.append({
                            'date': obs.get('date', ''),    # 数据日期
                            'value': value                  # 数据值
                        })
                    
                    economic_data[name] = processed_data
                    print(f"    ✅ {name} 数据获取成功")
                    break  # 成功获取，跳出重试循环
                    
                except Exception as e:
                    print(f"    尝试 {attempt + 1} 失败: {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2)  # 等待后重试
            
            # 如果某个指标获取失败，设为空列表
            if name not in economic_data:
                print(f"    ❌ {name} 最终获取失败")
                economic_data[name] = []
        
        print("✅ 经济指标数据获取完成")
        return economic_data
    
    def get_supported_symbols(self):
        """
        获取支持的股票代码列表
        
        这是一个简单的getter方法
        返回当前支持分析的所有股票代码
        """
        return self.stock_symbols.copy()  # 返回副本，避免外部修改原列表
    
    def validate_symbol(self, symbol):
        """
        验证股票代码是否支持
        
        参数：
        symbol (str): 要验证的股票代码
        
        返回：
        bool: True表示支持，False表示不支持
        """
        return symbol.upper() in [s.upper() for s in self.stock_symbols]

# 模块测试代码
# 只有直接运行这个文件时才会执行，被import时不会执行
if __name__ == "__main__":
    print("=== 数据收集模块测试 ===")
    
    # 创建数据收集器实例
    collector = MarketDataCollector()
    
    # 测试支持的股票列表
    print(f"支持的股票: {collector.get_supported_symbols()}")
    
    # 测试股票代码验证
    print(f"AAPL是否支持: {collector.validate_symbol('AAPL')}")
    print(f"INVALID是否支持: {collector.validate_symbol('INVALID')}")
    
    # 测试获取股票数据（使用小时间范围避免等待太久）
    print("\n--- 测试股票数据获取 ---")
    stock_data = collector.fetch_stock_data('AAPL', '1mo')  # 获取1个月数据
    
    if stock_data:
        df = stock_data['price_data']
        print(f"✅ 获取成功，数据行数: {len(df)}")
        print(f"最新收盘价: ${df['Close'].iloc[-1]:.2f}")
        print(f"技术指标列: {[col for col in df.columns if col.startswith(('MA', 'RSI', 'MACD'))]}")
    else:
        print("❌ 股票数据获取失败")
    
    # 测试新闻情感数据
    print("\n--- 测试新闻情感数据获取 ---")
    news_data = collector.fetch_news_sentiment('AAPL')
    print(f"获取到 {len(news_data)} 条新闻")
    
    # 测试经济指标数据  
    print("\n--- 测试经济指标数据获取 ---")
    economic_data = collector.fetch_economic_indicators()
    print(f"获取到 {len(economic_data)} 个经济指标")
    for name, data in economic_data.items():
        print(f"  {name}: {len(data)} 个数据点")
    
    print("\n=== 测试完成 ===")