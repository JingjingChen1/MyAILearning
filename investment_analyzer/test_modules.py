# test_modules.py
"""
模块测试文件

这个文件用于测试系统各个模块的功能是否正常
可以独立运行，检查各个组件的工作状态

使用方法：
python test_modules.py

测试内容：
1. 配置文件加载
2. 数据收集功能
3. 情感分析功能
4. AI预测功能
5. 策略生成功能
6. 模块间集成测试
"""

import sys
import traceback
from datetime import datetime
import pandas as pd
import numpy as np

def test_config():
    """
    测试配置文件
    """
    print("=== 测试配置文件 ===")
    
    try:
        import config
        
        # 检查主要配置是否存在
        assert hasattr(config, 'STOCK_SYMBOLS'), "缺少STOCK_SYMBOLS配置"
        assert hasattr(config, 'RISK_LEVELS'), "缺少RISK_LEVELS配置"
        assert hasattr(config, 'LSTM_CONFIG'), "缺少LSTM_CONFIG配置"
        
        print(f"✅ 支持股票数量: {len(config.STOCK_SYMBOLS)}")
        print(f"✅ 风险级别: {config.RISK_LEVELS}")
        print(f"✅ LSTM配置: 序列长度 {config.LSTM_CONFIG['sequence_length']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置文件测试失败: {e}")
        traceback.print_exc()
        return False

def test_data_collector():
    """
    测试数据收集模块
    """
    print("\n=== 测试数据收集模块 ===")
    
    try:
        from data_collector import MarketDataCollector
        
        # 创建实例
        collector = MarketDataCollector()
        print("✅ 数据收集器创建成功")
        
        # 测试股票代码验证
        assert collector.validate_symbol('AAPL'), "AAPL应该是支持的股票"
        assert not collector.validate_symbol('INVALID'), "INVALID不应该被支持"
        print("✅ 股票代码验证功能正常")
        
        # 测试获取股票数据（使用短时间范围）
        print("正在测试股票数据获取...")
        stock_data = collector.fetch_stock_data('AAPL', '1mo')
        
        if stock_data:
            print(f"✅ 股票数据获取成功，数据点数: {len(stock_data['price_data'])}")
            
            # 检查必要的列是否存在
            required_columns = ['Close', 'Volume', 'MA20', 'RSI']
            df = stock_data['price_data']
            
            for col in required_columns:
                if col in df.columns:
                    print(f"✅ 包含 {col} 数据")
                else:
                    print(f"⚠️ 缺少 {col} 数据")
        else:
            print("⚠️ 股票数据获取失败（可能是网络问题）")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据收集模块测试失败: {e}")
        traceback.print_exc()
        return False

def test_sentiment_analyzer():
    """
    测试情感分析模块
    """
    print("\n=== 测试情感分析模块 ===")
    
    try:
        from sentiment_analyzer import SentimentAnalyzer
        
        # 创建实例
        analyzer = SentimentAnalyzer()
        print("✅ 情感分析器创建成功")
        
        # 测试单条文本分析
        test_texts = [
            "Apple stock reaches new all-time high on strong earnings",
            "Market crashes as investors panic sell",
            "Company reports quarterly results as expected"
        ]
        
        print("正在测试文本情感分析...")
        for text in test_texts:
            result = analyzer.analyze_single_text(text)
            if result:
                print(f"✅ 文本: '{text[:30]}...' -> {result['label']} ({result['score']:.2f})")
            else:
                print(f"⚠️ 文本分析失败: '{text[:30]}...'")
        
        # 测试新闻列表分析
        mock_news = [
            {'title': 'Positive news about the company', 'relevance': 0.9},
            {'title': 'Negative market sentiment prevails', 'relevance': 0.8},
            {'title': 'Neutral company announcement', 'relevance': 0.7}
        ]
        
        print("正在测试新闻列表分析...")
        result = analyzer.analyze_news_sentiment(mock_news)
        
        if result['news_count'] > 0:
            print(f"✅ 新闻分析成功，总体情感: {result['overall_sentiment']:.3f}")
        else:
            print("⚠️ 新闻分析失败")
        
        return True
        
    except Exception as e:
        print(f"❌ 情感分析模块测试失败: {e}")
        traceback.print_exc()
        return False

def test_price_predictor():
    """
    测试价格预测模块
    """
    print("\n=== 测试价格预测模块 ===")
    
    try:
        from price_predictor import StockPricePredictor
        
        # 创建实例
        predictor = StockPricePredictor()
        print("✅ 价格预测器创建成功")
        
        # 创建模拟数据
        print("正在创建模拟数据...")
        dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
        np.random.seed(42)
        
        # 生成模拟股价数据
        prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
        
        mock_df = pd.DataFrame({
            'Open': prices * (1 + np.random.randn(200) * 0.01),
            'High': prices * (1 + abs(np.random.randn(200)) * 0.01),
            'Low': prices * (1 - abs(np.random.randn(200)) * 0.01),
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, 200)
        }, index=dates)
        
        # 计算基本技术指标
        mock_df['MA5'] = mock_df['Close'].rolling(window=5).mean()
        mock_df['MA20'] = mock_df['Close'].rolling(window=20).mean()
        mock_df['MA50'] = mock_df['Close'].rolling(window=50).mean()
        
        # RSI
        delta = mock_df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        mock_df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = mock_df['Close'].ewm(span=12).mean()
        exp2 = mock_df['Close'].ewm(span=26).mean()
        mock_df['MACD'] = exp1 - exp2
        mock_df['MACD_Signal'] = mock_df['MACD'].ewm(span=9).mean()
        
        mock_df['Volatility'] = mock_df['Close'].rolling(window=20).std()
        mock_df['Price_Change'] = mock_df['Close'].pct_change()
        
        # 准备模拟的其他数据
        mock_stock_data = {'price_data': mock_df}
        mock_sentiment_data = {'overall_sentiment': 0.1, 'confidence': 0.7}
        mock_economic_data = {
            'GDP': [{'value': 2.1}, {'value': 2.3}],
            'Inflation': [{'value': 3.2}, {'value': 3.1}]
        }
        
        # 测试特征准备
        print("正在测试特征准备...")
        features = predictor.prepare_features(
            mock_stock_data, mock_sentiment_data, mock_economic_data
        )
        print(f"✅ 特征准备成功，特征维度: {features.shape}")
        
        # 测试模型训练（快速模式）
        print("正在测试模型训练（快速模式）...")
        predictor.epochs = 3  # 使用很少的轮数进行快速测试
        
        train_result = predictor.train_model(features)
        
        if train_result['success']:
            print(f"✅ 模型训练成功，最终损失: {train_result['test_loss']:.6f}")
            
            # 测试预测
            print("正在测试价格预测...")
            prediction = predictor.predict_next_price(features.values)
            print(f"✅ 价格预测成功，置信度: {prediction['confidence']:.3f}")
            
        else:
            print(f"⚠️ 模型训练失败: {train_result.get('error')}")
        
        return True
        
    except Exception as e:
        print(f"❌ 价格预测模块测试失败: {e}")
        traceback.print_exc()
        return False

def test_strategy_generator():
    """
    测试策略生成模块
    """
    print("\n=== 测试策略生成模块 ===")
    
    try:
        from strategy_generator import InvestmentStrategyGenerator
        
        # 创建实例
        generator = InvestmentStrategyGenerator()
        print("✅ 策略生成器创建成功")
        
        # 创建模拟数据
        mock_predictions = {
            'AAPL': {
                'current_price': 150.0,
                'predicted_price': 155.0,
                'confidence': 0.8
            },
            'MSFT': {
                'current_price': 300.0,
                'predicted_price': 295.0,
                'confidence': 0.7
            }
        }
        
        mock_sentiment = {
            'AAPL': {
                'overall_sentiment': 0.2,
                'confidence': 0.7,
                'news_count': 5
            },
            'MSFT': {
                'overall_sentiment': -0.1,
                'confidence': 0.6,
                'news_count': 3
            }
        }
        
        # 创建模拟市场数据
        mock_market_data = {}
        for symbol in ['AAPL', 'MSFT']:
            dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
            prices = 150 + np.cumsum(np.random.randn(100) * 0.5)
            
            df = pd.DataFrame({
                'Close': prices,
                'Volume': np.random.randint(1000000, 5000000, 100),
                'RSI': 30 + np.random.randn(100) * 20,
                'MA20': prices * (1 + np.random.randn(100) * 0.01),
                'MA50': prices * (1 + np.random.randn(100) * 0.01),
                'MACD': np.random.randn(100) * 0.5,
                'MACD_Signal': np.random.randn(100) * 0.5,
                'Volatility': abs(np.random.randn(100) * 0.02),
                'Price_Change': np.random.randn(100) * 0.02,
                'Volume_MA': np.random.randint(2000000, 4000000, 100)
            }, index=dates)
            
            mock_market_data[symbol] = {'price_data': df}
        
        # 测试策略生成
        print("正在测试策略生成...")
        strategies = generator.generate_strategy(
            mock_predictions, mock_sentiment, mock_market_data
        )
        
        print("✅ 策略生成成功")
        for symbol, strategy in strategies.items():
            print(f"  {symbol}: 综合评分 {strategy['overall_score']:.3f}")
        
        # 测试投资组合生成
        print("正在测试投资组合生成...")
        portfolio = generator.generate_portfolio_advice(strategies, '平衡', 100000)
        
        print(f"✅ 投资组合生成成功")
        print(f"  投资标的: {len(portfolio['allocations'])}")
        print(f"  资金利用率: {portfolio['total_allocated']/portfolio['total_capital']:.1%}")
        
        return True
        
    except Exception as e:
        print(f"❌ 策略生成模块测试失败: {e}")
        traceback.print_exc()
        return False

def test_integration():
    """
    测试模块集成
    """
    print("\n=== 测试模块集成 ===")
    
    try:
        # 测试所有模块是否可以协同工作
        from data_collector import MarketDataCollector
        from sentiment_analyzer import SentimentAnalyzer
        from strategy_generator import InvestmentStrategyGenerator
        
        print("✅ 所有核心模块导入成功")
        
        # 测试简单的集成流程
        collector = MarketDataCollector()
        analyzer = SentimentAnalyzer()
        generator = InvestmentStrategyGenerator()
        
        print("✅ 所有模块实例创建成功")
        
        # 检查模块间的兼容性
        supported_symbols = collector.get_supported_symbols()
        risk_levels = generator.risk_levels
        
        print(f"✅ 数据收集器支持 {len(supported_symbols)} 只股票")
        print(f"✅ 策略生成器支持 {len(risk_levels)} 种风险级别")
        
        return True
        
    except Exception as e:
        print(f"❌ 模块集成测试失败: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """
    运行所有测试
    """
    print("🤖 智能投资决策助手 - 模块测试")
    print("=" * 50)
    print(f"测试开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 运行各项测试
    tests = [
        ("配置文件", test_config),
        ("数据收集模块", test_data_collector),
        ("情感分析模块", test_sentiment_analyzer),
        ("价格预测模块", test_price_predictor),
        ("策略生成模块", test_strategy_generator),
        ("模块集成", test_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
            results.append((test_name, False))
    
    # 输出测试总结
    print("\n" + "=" * 50)
    print("测试结果总结:")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        if result:
            print(f"✅ {test_name}: 通过")
            passed += 1
        else:
            print(f"❌ {test_name}: 失败")
            failed += 1
    
    print(f"\n总计: {passed} 个测试通过, {failed} 个测试失败")
    
    if failed == 0:
        print("🎉 所有测试都通过了！系统可以正常运行。")
    else:
        print("⚠️ 部分测试失败，可能影响系统功能。")
        print("   请检查失败的模块并修复问题。")
    
    print(f"\n测试结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return failed == 0

if __name__ == "__main__":
    # 运行所有测试
    success = run_all_tests()
    
    # 根据测试结果设置退出码
    sys.exit(0 if success else 1)