# web_interface.py
"""
Web界面模块

这个模块的作用：
1. 使用Streamlit创建用户友好的Web界面
2. 整合所有功能模块，提供完整的用户体验
3. 处理用户输入，展示分析结果
4. 提供交互式图表和数据可视化

什么是Streamlit？
- Python的Web应用开发框架
- 特别适合数据科学和机器学习应用
- 可以快速将Python脚本转换为Web应用
- 内置许多数据可视化组件
"""

import streamlit as st          # Streamlit主框架
import plotly.graph_objects as go  # 交互式图表库
import plotly.express as px       # 快速图表创建
import pandas as pd              # 数据处理
import numpy as np               # 数学计算
from datetime import datetime, timedelta  # 日期时间处理
import warnings                  # 警告管理
import traceback                # 错误追踪

# 导入我们自己的模块
from data_collector import MarketDataCollector
from sentiment_analyzer import SentimentAnalyzer
from price_predictor import StockPricePredictor
from strategy_generator import InvestmentStrategyGenerator

# 导入配置
from config import (
    STOCK_SYMBOLS, RISK_LEVELS, UI_CONFIG, 
    ERROR_MESSAGES, SUCCESS_MESSAGES
)

# 忽略警告
warnings.filterwarnings('ignore')

class WebInterface:
    """
    Web界面类
    
    负责创建和管理整个应用的用户界面
    协调各个功能模块的交互
    """
    
    def __init__(self):
        """
        初始化Web界面
        
        创建各个功能模块的实例
        设置界面基本配置
        """
        print("正在初始化Web界面...")
        
        # 初始化各个功能模块
        # 使用try-except确保即使某个模块初始化失败，应用也能继续运行
        try:
            self.data_collector = MarketDataCollector()
            print("✅ 数据收集器初始化成功")
        except Exception as e:
            print(f"❌ 数据收集器初始化失败: {e}")
            self.data_collector = None
        
        try:
            self.sentiment_analyzer = SentimentAnalyzer()
            print("✅ 情感分析器初始化成功")
        except Exception as e:
            print(f"❌ 情感分析器初始化失败: {e}")
            self.sentiment_analyzer = None
        
        try:
            self.predictor = StockPricePredictor()
            print("✅ 价格预测器初始化成功")
        except Exception as e:
            print(f"❌ 价格预测器初始化失败: {e}")
            self.predictor = None
        
        try:
            self.strategy_generator = InvestmentStrategyGenerator()
            print("✅ 策略生成器初始化成功")
        except Exception as e:
            print(f"❌ 策略生成器初始化失败: {e}")
            self.strategy_generator = None
        
        # 界面状态管理
        # Streamlit的session_state用于在用户交互之间保持数据
        self._initialize_session_state()
        
        print("✅ Web界面初始化完成")
    
    def _initialize_session_state(self):
        """
        初始化Streamlit的会话状态
        
        session_state是Streamlit提供的状态管理机制
        用于在页面重新加载时保持数据
        """
        
        # 分析结果缓存
        if 'market_data' not in st.session_state:
            st.session_state.market_data = {}
        
        if 'sentiment_data' not in st.session_state:
            st.session_state.sentiment_data = {}
        
        if 'predictions' not in st.session_state:
            st.session_state.predictions = {}
        
        if 'strategies' not in st.session_state:
            st.session_state.strategies = {}
        
        if 'portfolio' not in st.session_state:
            st.session_state.portfolio = {}
        
        # 分析状态标记
        if 'analysis_completed' not in st.session_state:
            st.session_state.analysis_completed = False
        
        if 'last_analysis_time' not in st.session_state:
            st.session_state.last_analysis_time = None
        
        # 错误信息
        if 'error_messages' not in st.session_state:
            st.session_state.error_messages = []
    
    def run_app(self):
        """
        运行Streamlit应用
        
        这是应用的主要入口点
        设置页面配置和布局，处理用户交互
        """
        
        # 设置页面配置
        # 这必须是第一个Streamlit命令
        st.set_page_config(
            page_title=UI_CONFIG['page_title'],    # 浏览器标签页标题
            page_icon=UI_CONFIG['page_icon'],      # 浏览器标签页图标
            layout=UI_CONFIG['layout'],            # 页面布局（wide=宽屏）
            initial_sidebar_state="expanded"       # 侧边栏初始状态
        )
        
        # 主标题和说明
        st.title("🤖 智能投资决策助手")
        st.markdown("""
        这是一个基于AI的股票投资分析系统，整合了：
        - 📈 **技术分析**：移动平均线、RSI、MACD等技术指标
        - 🎭 **情感分析**：基于新闻的市场情感评估
        - 🧠 **AI预测**：LSTM神经网络价格预测
        - 💡 **智能策略**：个性化投资建议生成
        
        ---
        """)
        
        # 检查模块可用性
        self._check_module_availability()
        
        # 创建侧边栏控制面板
        self._create_sidebar()
        
        # 根据分析状态显示不同内容
        if st.session_state.analysis_completed:
            self._display_analysis_results()
        else:
            self._display_welcome_screen()
    
    def _check_module_availability(self):
        """
        检查各功能模块的可用性
        
        如果某些模块初始化失败，显示相应的警告信息
        """
        unavailable_modules = []
        
        if self.data_collector is None:
            unavailable_modules.append("数据收集")
        if self.sentiment_analyzer is None:
            unavailable_modules.append("情感分析")
        if self.predictor is None:
            unavailable_modules.append("AI预测")
        if self.strategy_generator is None:
            unavailable_modules.append("策略生成")
        
        if unavailable_modules:
            st.warning(f"⚠️ 以下模块不可用：{', '.join(unavailable_modules)}")
            st.info("某些功能可能受限，但您仍可以使用其他可用功能。")
    
    def _create_sidebar(self):
        """
        创建侧边栏控制面板
        
        包含用户输入控件和分析触发按钮
        """
        st.sidebar.title("📊 控制面板")
        
        # 1. 股票选择
        st.sidebar.subheader("📈 选择股票")
        selected_stocks = st.sidebar.multiselect(
            "请选择要分析的股票：",
            options=STOCK_SYMBOLS,                    # 可选选项
            default=UI_CONFIG['default_stocks'],      # 默认选中的股票
            help="可以选择多只股票进行对比分析"        # 帮助提示
        )
        
        # 2. 风险偏好设置
        st.sidebar.subheader("⚖️ 风险偏好")
        risk_preference = st.sidebar.selectbox(
            "请选择您的风险偏好：",
            options=RISK_LEVELS,
            index=RISK_LEVELS.index(UI_CONFIG['default_risk']),  # 默认选中项的索引
            help="不同风险偏好会影响投资建议的激进程度"
        )
        
        # 3. 分析参数设置
        st.sidebar.subheader("🔧 分析参数")
        
        # 数据时间范围
        time_period = st.sidebar.selectbox(
            "数据时间范围：",
            options=['1mo', '3mo', '6mo', '1y', '2y'],
            index=3,  # 默认选择1年
            help="更长的时间范围提供更多历史数据，但分析耗时更长"
        )
        
        # AI训练设置
        with st.sidebar.expander("🧠 AI训练设置", expanded=False):
            quick_mode = st.checkbox(
                "快速模式",
                value=True,
                help="快速模式使用较少的训练轮数，分析速度更快但精度略低"
            )
            
            if quick_mode:
                train_epochs = 10
                st.info("快速模式：10轮训练")
            else:
                train_epochs = st.slider("训练轮数", 20, 100, 50)
        
        # 4. 开始分析按钮
        st.sidebar.markdown("---")
        
        # 检查输入有效性
        can_analyze = (
            len(selected_stocks) > 0 and
            self.data_collector is not None
        )
        
        if not can_analyze:
            if len(selected_stocks) == 0:
                st.sidebar.error("❌ 请至少选择一只股票")
            elif self.data_collector is None:
                st.sidebar.error("❌ 数据收集模块不可用")
        
        # 分析按钮
        if st.sidebar.button(
            "🚀 开始智能分析",
            disabled=not can_analyze,
            help="点击开始完整的投资分析流程"
        ):
            # 清除之前的结果和错误
            st.session_state.analysis_completed = False
            st.session_state.error_messages = []
            
            # 更新AI训练参数
            if self.predictor:
                self.predictor.epochs = train_epochs
            
            # 开始分析
            self._run_complete_analysis(
                selected_stocks, risk_preference, time_period
            )
        
        # 5. 重置按钮
        if st.sidebar.button("🔄 重置分析"):
            self._reset_analysis()
            st.rerun()  # 重新加载页面
        
        # 6. 显示上次分析时间
        if st.session_state.last_analysis_time:
            st.sidebar.markdown("---")
            st.sidebar.caption(f"上次分析：{st.session_state.last_analysis_time}")
    
    def _display_welcome_screen(self):
        """
        显示欢迎屏幕
        
        当用户还没有进行分析时显示的界面
        """
        # 功能介绍
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔍 分析功能")
            st.markdown("""
            **技术分析**
            - 移动平均线 (MA5/20/50)
            - 相对强弱指标 (RSI)
            - MACD指标
            - 布林带
            - 成交量分析
            
            **AI智能预测**
            - LSTM神经网络
            - 多维特征工程
            - 时间序列预测
            - 置信度评估
            """)
        
        with col2:
            st.subheader("📊 提供建议")
            st.markdown("""
            **情感分析**
            - 新闻情感识别
            - 市场情绪评估
            - 社交媒体监控
            - 综合情感评分
            
            **投资策略**
            - 个性化风险匹配
            - 仓位管理建议
            - 止盈止损设置
            - 投资组合优化
            """)
        
        # 使用说明
        st.subheader("📖 使用说明")
        st.markdown("""
        1. **选择股票**：在左侧面板选择您感兴趣的股票（可多选）
        2. **设置风险偏好**：根据您的风险承受能力选择相应级别
        3. **调整参数**：可选择数据时间范围和AI训练模式
        4. **开始分析**：点击"开始智能分析"按钮，等待分析完成
        5. **查看结果**：系统将显示详细的分析报告和投资建议
        """)
        
        # 风险提示
        st.warning("""
        ⚠️ **重要提示**
        
        本系统提供的分析和建议仅供参考，不构成投资建议。投资有风险，入市需谨慎。
        请在做出投资决策前，充分了解相关风险，并根据自身情况谨慎决策。
        """)
        
        # 示例展示
        st.subheader("📈 示例图表")
        
        # 创建示例数据和图表
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=prices,
            mode='lines',
            name='股价走势',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title="示例：股价走势图",
            xaxis_title="日期",
            yaxis_title="价格 ($)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _run_complete_analysis(self, symbols, risk_preference, time_period):
        """
        运行完整的分析流程
        
        这是系统的核心方法，协调所有分析步骤
        
        参数：
        symbols (list): 要分析的股票代码列表
        risk_preference (str): 风险偏好
        time_period (str): 数据时间范围
        """
        
        # 创建进度条
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # 第1步：数据收集 (0-30%)
            status_text.text("📥 正在收集市场数据...")
            market_data, economic_data = self._collect_market_data(
                symbols, time_period, progress_bar, 0, 30
            )
            
            if not market_data:
                st.error("❌ 未能获取任何股票数据，请检查网络连接或稍后重试")
                return
            
            # 第2步：情感分析 (30-50%)
            status_text.text("🎭 正在分析市场情感...")
            sentiment_data = self._analyze_market_sentiment(
                symbols, progress_bar, 30, 50
            )
            
            # 第3步：AI价格预测 (50-80%)
            status_text.text("🧠 正在进行AI价格预测...")
            predictions = self._predict_stock_prices(
                market_data, sentiment_data, economic_data,
                progress_bar, 50, 80
            )
            
            # 第4步：策略生成 (80-100%)
            status_text.text("💡 正在生成投资策略...")
            strategies = self._generate_investment_strategies(
                predictions, sentiment_data, market_data, risk_preference,
                progress_bar, 80, 100
            )
            
            # 保存结果到session_state
            st.session_state.market_data = market_data
            st.session_state.sentiment_data = sentiment_data
            st.session_state.predictions = predictions
            st.session_state.strategies = strategies
            st.session_state.analysis_completed = True
            st.session_state.last_analysis_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 完成
            progress_bar.progress(100)
            status_text.text("✅ 分析完成！")
            
            # 短暂显示完成信息后清除
            import time
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            # 重新加载页面显示结果
            st.rerun()
            
        except Exception as e:
            # 错误处理
            progress_bar.empty()
            status_text.empty()
            
            error_msg = f"分析过程中出现错误：{str(e)}"
            st.error(error_msg)
            
            # 显示详细错误信息（仅在开发模式下）
            with st.expander("🔍 错误详情（开发者信息）"):
                st.code(traceback.format_exc())
            
            st.session_state.error_messages.append(error_msg)
    
    def _collect_market_data(self, symbols, time_period, progress_bar, start_pct, end_pct):
        """
        收集市场数据
        
        参数：
        symbols (list): 股票代码列表
        time_period (str): 时间范围
        progress_bar: 进度条对象
        start_pct, end_pct: 进度百分比范围
        
        返回：
        tuple: (market_data, economic_data)
        """
        
        market_data = {}
        total_symbols = len(symbols)
        
        # 获取股票数据
        for i, symbol in enumerate(symbols):
            try:
                # 更新进度
                current_progress = start_pct + (i / total_symbols) * (end_pct - start_pct - 5)
                progress_bar.progress(int(current_progress))
                
                # 获取股票数据
                stock_data = self.data_collector.fetch_stock_data(symbol, time_period)
                
                if stock_data:
                    market_data[symbol] = stock_data
                    print(f"✅ {symbol} 数据获取成功")
                else:
                    print(f"❌ {symbol} 数据获取失败")
                    st.warning(f"⚠️ {symbol} 数据获取失败，将跳过该股票")
                
            except Exception as e:
                print(f"❌ {symbol} 数据获取异常: {e}")
                st.warning(f"⚠️ {symbol} 数据获取异常：{str(e)}")
        
        # 获取经济数据
        try:
            progress_bar.progress(end_pct - 5)
            economic_data = self.data_collector.fetch_economic_indicators()
        except Exception as e:
            print(f"❌ 经济数据获取失败: {e}")
            st.warning("⚠️ 经济指标数据获取失败，将使用默认值")
            economic_data = {}
        
        progress_bar.progress(end_pct)
        return market_data, economic_data
    
    def _analyze_market_sentiment(self, symbols, progress_bar, start_pct, end_pct):
        """
        分析市场情感
        
        参数：
        symbols (list): 股票代码列表
        progress_bar: 进度条对象
        start_pct, end_pct: 进度百分比范围
        
        返回：
        dict: 情感分析结果
        """
        
        sentiment_data = {}
        
        if self.sentiment_analyzer is None:
            # 如果情感分析器不可用，返回默认值
            for symbol in symbols:
                sentiment_data[symbol] = {
                    'overall_sentiment': 0.0,
                    'confidence': 0.0,
                    'news_count': 0,
                    'error': '情感分析模块不可用'
                }
            progress_bar.progress(end_pct)
            return sentiment_data
        
        total_symbols = len(symbols)
        
        for i, symbol in enumerate(symbols):
            try:
                # 更新进度
                current_progress = start_pct + (i / total_symbols) * (end_pct - start_pct)
                progress_bar.progress(int(current_progress))
                
                # 获取新闻数据
                news_data = self.data_collector.fetch_news_sentiment(symbol)
                
                # 分析情感
                sentiment_result = self.sentiment_analyzer.analyze_news_sentiment(news_data)
                sentiment_data[symbol] = sentiment_result
                
                print(f"✅ {symbol} 情感分析完成")
                
            except Exception as e:
                print(f"❌ {symbol} 情感分析失败: {e}")
                sentiment_data[symbol] = {
                    'overall_sentiment': 0.0,
                    'confidence': 0.0,
                    'news_count': 0,
                    'error': str(e)
                }
        
        progress_bar.progress(end_pct)
        return sentiment_data
    
    def _predict_stock_prices(self, market_data, sentiment_data, economic_data, 
                            progress_bar, start_pct, end_pct):
        """
        预测股票价格
        
        参数：
        market_data (dict): 市场数据
        sentiment_data (dict): 情感数据
        economic_data (dict): 经济数据
        progress_bar: 进度条对象
        start_pct, end_pct: 进度百分比范围
        
        返回：
        dict: 价格预测结果
        """
        
        predictions = {}
        
        if self.predictor is None:
            # 如果预测器不可用，返回简单的线性预测
            for symbol in market_data.keys():
                current_price = market_data[symbol]['price_data']['Close'].iloc[-1]
                # 简单线性预测：基于最近5天的平均变化
                recent_changes = market_data[symbol]['price_data']['Price_Change'].tail(5).mean()
                predicted_price = current_price * (1 + recent_changes)
                
                predictions[symbol] = {
                    'current_price': current_price,
                    'predicted_price': predicted_price,
                    'confidence': 0.3,  # 低置信度
                    'method': 'linear_fallback',
                    'error': 'AI预测模块不可用，使用简单线性预测'
                }
            
            progress_bar.progress(end_pct)
            return predictions
        
        symbols = list(market_data.keys())
        total_symbols = len(symbols)
        
        for i, symbol in enumerate(symbols):
            try:
                # 更新进度
                current_progress = start_pct + (i / total_symbols) * (end_pct - start_pct)
                progress_bar.progress(int(current_progress))
                
                # 准备特征数据
                features = self.predictor.prepare_features(
                    market_data[symbol],
                    sentiment_data[symbol],
                    economic_data
                )
                
                # 训练模型
                train_result = self.predictor.train_model(features)
                
                if train_result['success']:
                    # 预测下一个价格
                    prediction_result = self.predictor.predict_next_price(features.values)
                    
                    current_price = market_data[symbol]['price_data']['Close'].iloc[-1]
                    
                    predictions[symbol] = {
                        'current_price': current_price,
                        'predicted_price': prediction_result['predicted_price'],
                        'confidence': prediction_result['confidence'],
                        'method': 'LSTM',
                        'train_metrics': {
                            'mse': train_result.get('mse'),
                            'mae': train_result.get('test_mae'),
                            'mape': train_result.get('mape')
                        }
                    }
                    
                    print(f"✅ {symbol} AI预测完成")
                
                else:
                    raise Exception(train_result.get('error', '训练失败'))
                
            except Exception as e:
                print(f"❌ {symbol} AI预测失败: {e}")
                
                # 回退到简单预测
                current_price = market_data[symbol]['price_data']['Close'].iloc[-1]
                recent_changes = market_data[symbol]['price_data']['Price_Change'].tail(5).mean()
                predicted_price = current_price * (1 + recent_changes)
                
                predictions[symbol] = {
                    'current_price': current_price,
                    'predicted_price': predicted_price,
                    'confidence': 0.2,
                    'method': 'fallback',
                    'error': str(e)
                }
        
        progress_bar.progress(end_pct)
        return predictions
    
    def _generate_investment_strategies(self, predictions, sentiment_data, market_data, 
                                      risk_preference, progress_bar, start_pct, end_pct):
        """
        生成投资策略
        
        参数：
        predictions (dict): 价格预测结果
        sentiment_data (dict): 情感数据
        market_data (dict): 市场数据
        risk_preference (str): 风险偏好
        progress_bar: 进度条对象
        start_pct, end_pct: 进度百分比范围
        
        返回：
        dict: 投资策略结果
        """
        
        if self.strategy_generator is None:
            # 如果策略生成器不可用，返回简单策略
            strategies = {}
            for symbol in predictions.keys():
                expected_return = (predictions[symbol]['predicted_price'] - 
                                 predictions[symbol]['current_price']) / predictions[symbol]['current_price']
                
                simple_action = '买入' if expected_return > 0.02 else '持有' if expected_return > -0.02 else '卖出'
                
                strategies[symbol] = {
                    'symbol': symbol,
                    'overall_score': expected_return,
                    'recommendations': {
                        risk_preference: {
                            'action': simple_action,
                            'position_size': 0.1 if simple_action == '买入' else 0,
                            'reasoning': ['基于简单价格预测的建议'],
                            'confidence_level': '低'
                        }
                    },
                    'error': '策略生成模块不可用，使用简化策略'
                }
            
            progress_bar.progress(end_pct)
            return strategies
        
        try:
            progress_bar.progress(start_pct + 10)
            
            # 生成策略
            strategies = self.strategy_generator.generate_strategy(
                predictions, sentiment_data, market_data
            )
            
            progress_bar.progress(start_pct + 20)
            
            # 生成投资组合建议
            portfolio = self.strategy_generator.generate_portfolio_advice(
                strategies, risk_preference, 100000  # 假设10万资金
            )
            
            # 将投资组合信息保存到session_state
            st.session_state.portfolio = portfolio
            
            progress_bar.progress(end_pct)
            
            print("✅ 投资策略生成完成")
            return strategies
            
        except Exception as e:
            print(f"❌ 策略生成失败: {e}")
            
            # 返回错误策略
            strategies = {}
            for symbol in predictions.keys():
                strategies[symbol] = {
                    'symbol': symbol,
                    'error': True,
                    'error_message': str(e),
                    'recommendations': {
                        risk_preference: {
                            'action': '分析失败',
                            'position_size': 0,
                            'reasoning': [f'策略生成失败: {str(e)}'],
                            'confidence_level': '无'
                        }
                    }
                }
            
            progress_bar.progress(end_pct)
            return strategies
    
    def _display_analysis_results(self):
        """
        显示分析结果
        
        创建多个标签页展示不同类型的分析结果
        """
        st.success("✅ 分析完成！以下是详细的分析报告：")
        
        # 创建标签页
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 市场概览", "📈 技术分析", "🎭 情感分析", 
            "🤖 AI预测", "💡 投资建议"
        ])
        
        with tab1:
            self._display_market_overview()
        
        with tab2:
            self._display_technical_analysis()
        
        with tab3:
            self._display_sentiment_analysis()
        
        with tab4:
            self._display_ai_predictions()
        
        with tab5:
            self._display_investment_advice()
    
    def _display_market_overview(self):
        """
        显示市场概览
        """
        st.subheader("📊 市场数据概览")
        
        market_data = st.session_state.market_data
        
        if not market_data:
            st.warning("暂无市场数据")
            return
        
        # 创建概览表格
        overview_data = []
        for symbol, data in market_data.items():
            price_data = data['price_data']
            latest = price_data.iloc[-1]
            
            # 计算一些基本指标
            daily_change = latest['Price_Change'] if 'Price_Change' in price_data.columns else 0
            volume = latest['Volume']
            rsi = latest['RSI'] if 'RSI' in price_data.columns else None
            
            overview_data.append({
                '股票代码': symbol,
                '当前价格': f"${latest['Close']:.2f}",
                '日涨跌幅': f"{daily_change:.2%}",
                '成交量': f"{volume:,.0f}",
                'RSI': f"{rsi:.1f}" if rsi is not None else "N/A",
                '数据天数': len(price_data)
            })
        
        overview_df = pd.DataFrame(overview_data)
        st.dataframe(overview_df, use_container_width=True)
        
        # 显示价格走势图
        st.subheader("📈 价格走势对比")
        
        fig = go.Figure()
        
        for symbol, data in market_data.items():
            price_data = data['price_data'].tail(60)  # 显示最近60天
            
            fig.add_trace(go.Scatter(
                x=price_data.index,
                y=price_data['Close'],
                mode='lines',
                name=symbol,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title="股价走势对比（最近60天）",
            xaxis_title="日期",
            yaxis_title="价格 ($)",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_technical_analysis(self):
        """
        显示技术分析
        """
        st.subheader("📈 技术指标分析")
        
        market_data = st.session_state.market_data
        
        if not market_data:
            st.warning("暂无技术分析数据")
            return
        
        # 让用户选择要查看的股票
        selected_symbol = st.selectbox(
            "选择股票查看详细技术分析：",
            options=list(market_data.keys())
        )
        
        if selected_symbol:
            self._display_single_stock_technical_analysis(
                selected_symbol, market_data[selected_symbol]
            )
    
    def _display_single_stock_technical_analysis(self, symbol, stock_data):
        """
        显示单只股票的技术分析
        """
        st.subheader(f"{symbol} 技术分析")
        
        price_data = stock_data['price_data'].tail(100)  # 最近100天
        
        # 1. K线图和移动平均线
        fig = go.Figure()
        
        # 添加K线图
        fig.add_trace(go.Candlestick(
            x=price_data.index,
            open=price_data['Open'],
            high=price_data['High'],
            low=price_data['Low'],
            close=price_data['Close'],
            name='K线'
        ))
        
        # 添加移动平均线
        if 'MA20' in price_data.columns:
            fig.add_trace(go.Scatter(
                x=price_data.index,
                y=price_data['MA20'],
                mode='lines',
                name='MA20',
                line=dict(color='red', width=1)
            ))
        
        if 'MA50' in price_data.columns:
            fig.add_trace(go.Scatter(
                x=price_data.index,
                y=price_data['MA50'],
                mode='lines',
                name='MA50',
                line=dict(color='blue', width=1)
            ))
        
        fig.update_layout(
            title=f"{symbol} K线图和移动平均线",
            xaxis_title="日期",
            yaxis_title="价格 ($)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 2. RSI指标
        if 'RSI' in price_data.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(
                    x=price_data.index,
                    y=price_data['RSI'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='purple', width=2)
                ))
                
                # 添加超买超卖线
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="超买线")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="超卖线")
                
                fig_rsi.update_layout(
                    title="RSI指标",
                    xaxis_title="日期",
                    yaxis_title="RSI",
                    height=300
                )
                
                st.plotly_chart(fig_rsi, use_container_width=True)
            
            # 3. MACD指标
            with col2:
                if 'MACD' in price_data.columns:
                    fig_macd = go.Figure()
                    fig_macd.add_trace(go.Scatter(
                        x=price_data.index,
                        y=price_data['MACD'],
                        mode='lines',
                        name='MACD',
                        line=dict(color='blue', width=2)
                    ))
                    
                    if 'MACD_Signal' in price_data.columns:
                        fig_macd.add_trace(go.Scatter(
                            x=price_data.index,
                            y=price_data['MACD_Signal'],
                            mode='lines',
                            name='信号线',
                            line=dict(color='red', width=1)
                        ))
                    
                    fig_macd.update_layout(
                        title="MACD指标",
                        xaxis_title="日期",
                        yaxis_title="MACD",
                        height=300
                    )
                    
                    st.plotly_chart(fig_macd, use_container_width=True)
        
        # 4. 技术指标数值表
        st.subheader("最新技术指标数值")
        
        latest = price_data.iloc[-1]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'RSI' in price_data.columns:
                rsi_value = latest['RSI']
                rsi_status = "超买" if rsi_value > 70 else "超卖" if rsi_value < 30 else "正常"
                st.metric("RSI", f"{rsi_value:.1f}", delta=rsi_status)
        
        with col2:
            if 'Volatility' in price_data.columns:
                volatility = latest['Volatility']
                st.metric("波动率", f"{volatility:.3f}")
        
        with col3:
            if 'Price_Change' in price_data.columns:
                change = latest['Price_Change'] * 100
                st.metric("日变化", f"{change:.2f}%")
        
        with col4:
            volume = latest['Volume']
            st.metric("成交量", f"{volume:,.0f}")
    
    def _display_sentiment_analysis(self):
        """
        显示情感分析结果
        """
        st.subheader("🎭 市场情感分析")
        
        sentiment_data = st.session_state.sentiment_data
        
        if not sentiment_data:
            st.warning("暂无情感分析数据")
            return
        
        # 1. 情感总览
        sentiment_overview = []
        for symbol, data in sentiment_data.items():
            if 'error' not in data:
                sentiment_overview.append({
                    '股票代码': symbol,
                    '总体情感': f"{data['overall_sentiment']:.3f}",
                    '置信度': f"{data['confidence']:.2f}",
                    '新闻数量': data['news_count'],
                    '主导情感': data.get('dominant_sentiment', 'unknown')
                })
        
        if sentiment_overview:
            st.subheader("情感分析总览")
            sentiment_df = pd.DataFrame(sentiment_overview)
            st.dataframe(sentiment_df, use_container_width=True)
            
            # 2. 情感分布图
            st.subheader("情感分布可视化")
            
            symbols = [item['股票代码'] for item in sentiment_overview]
            sentiments = [float(item['总体情感']) for item in sentiment_overview]
            confidences = [float(item['置信度']) for item in sentiment_overview]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=symbols,
                y=sentiments,
                name='情感得分',
                marker_color=['green' if s > 0 else 'red' if s < 0 else 'gray' for s in sentiments]
            ))
            
            fig.update_layout(
                title="各股票情感得分对比",
                xaxis_title="股票代码",
                yaxis_title="情感得分",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 3. 详细新闻分析
            st.subheader("详细新闻分析")
            
            selected_symbol = st.selectbox(
                "选择股票查看详细新闻分析：",
                options=symbols
            )
            
            if selected_symbol:
                symbol_sentiment = sentiment_data[selected_symbol]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("总体情感", f"{symbol_sentiment['overall_sentiment']:.3f}")
                    st.metric("分析置信度", f"{symbol_sentiment['confidence']:.2f}")
                
                with col2:
                    st.metric("新闻数量", symbol_sentiment['news_count'])
                    st.metric("主导情感", symbol_sentiment.get('dominant_sentiment', 'unknown'))
                
                # 显示情感分布饼图
                if 'sentiment_distribution' in symbol_sentiment:
                    dist = symbol_sentiment['sentiment_distribution']
                    
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=list(dist.keys()),
                        values=list(dist.values()),
                        title=f"{selected_symbol} 情感分布"
                    )])
                    
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                # 显示详细新闻（如果有）
                if 'detailed_scores' in symbol_sentiment and symbol_sentiment['detailed_scores']:
                    st.subheader("新闻详情")
                    
                    news_data = []
                    for news in symbol_sentiment['detailed_scores'][:5]:  # 显示前5条
                        news_data.append({
                            '新闻标题': news['text'][:100] + '...' if len(news['text']) > 100 else news['text'],
                            '情感标签': news['label'],
                            '置信度': f"{news['score']:.2f}",
                            '相关性': f"{news['relevance']:.2f}"
                        })
                    
                    if news_data:
                        news_df = pd.DataFrame(news_data)
                        st.dataframe(news_df, use_container_width=True)
        
        else:
            st.warning("所有股票的情感分析都失败了")
            
            # 显示错误信息
            st.subheader("错误详情")
            for symbol, data in sentiment_data.items():
                if 'error' in data:
                    st.error(f"{symbol}: {data['error']}")
    
    def _display_ai_predictions(self):
        """
        显示AI预测结果
        """
        st.subheader("🤖 AI价格预测")
        
        predictions = st.session_state.predictions
        market_data = st.session_state.market_data
        
        if not predictions:
            st.warning("暂无AI预测数据")
            return
        
        # 1. 预测总览表
        prediction_overview = []
        for symbol, pred in predictions.items():
            expected_return = (pred['predicted_price'] - pred['current_price']) / pred['current_price']
            
            prediction_overview.append({
                '股票代码': symbol,
                '当前价格': f"${pred['current_price']:.2f}",
                '预测价格': f"${pred['predicted_price']:.2f}",
                '预期收益': f"{expected_return:.2%}",
                '置信度': f"{pred['confidence']:.2f}",
                '预测方法': pred.get('method', 'unknown')
            })
        
        st.subheader("AI预测总览")
        pred_df = pd.DataFrame(prediction_overview)
        st.dataframe(pred_df, use_container_width=True)
        
        # 2. 预测结果可视化
        st.subheader("预测结果可视化")
        
        symbols = [item['股票代码'] for item in prediction_overview]
        current_prices = [pred['current_price'] for pred in predictions.values()]
        predicted_prices = [pred['predicted_price'] for pred in predictions.values()]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=symbols,
            y=current_prices,
            name='当前价格',
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            x=symbols,
            y=predicted_prices,
            name='预测价格',
            marker_color='orange'
        ))
        
        fig.update_layout(
            title="当前价格 vs 预测价格",
            xaxis_title="股票代码",
            yaxis_title="价格 ($)",
            barmode='group',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 3. 预期收益率图
        expected_returns = [(pred['predicted_price'] - pred['current_price']) / pred['current_price'] 
                           for pred in predictions.values()]
        
        fig_returns = go.Figure()
        fig_returns.add_trace(go.Bar(
            x=symbols,
            y=[r * 100 for r in expected_returns],  # 转换为百分比
            name='预期收益率',
            marker_color=['green' if r > 0 else 'red' for r in expected_returns]
        ))
        
        fig_returns.update_layout(
            title="预期收益率分布",
            xaxis_title="股票代码",
            yaxis_title="预期收益率 (%)",
            height=400
        )
        
        st.plotly_chart(fig_returns, use_container_width=True)
        
        # 4. 模型性能指标（如果有）
        st.subheader("模型性能指标")
        
        performance_data = []
        for symbol, pred in predictions.items():
            if 'train_metrics' in pred and pred['train_metrics']:
                metrics = pred['train_metrics']
                performance_data.append({
                    '股票代码': symbol,
                    '均方误差(MSE)': f"{metrics.get('mse', 'N/A')}",
                    '平均绝对误差(MAE)': f"{metrics.get('mae', 'N/A')}",
                    '平均绝对百分比误差(MAPE)': f"{metrics.get('mape', 'N/A')}%",
                    '预测方法': pred.get('method', 'unknown')
                })
        
        if performance_data:
            perf_df = pd.DataFrame(performance_data)
            st.dataframe(perf_df, use_container_width=True)
        else:
            st.info("暂无详细的模型性能指标")
        
        # 5. 错误信息显示
        error_predictions = {symbol: pred for symbol, pred in predictions.items() if 'error' in pred}
        if error_predictions:
            st.subheader("预测错误信息")
            for symbol, pred in error_predictions.items():
                st.error(f"{symbol}: {pred['error']}")
    
    def _display_investment_advice(self):
        """
        显示投资建议
        """
        st.subheader("💡 个性化投资建议")
        
        strategies = st.session_state.strategies
        portfolio = st.session_state.portfolio
        
        if not strategies:
            st.warning("暂无投资策略数据")
            return
        
        # 1. 投资组合概览
        if portfolio:
            st.subheader("🎯 投资组合建议")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("总资金", f"${portfolio['total_capital']:,.0f}")
            
            with col2:
                st.metric("已分配资金", f"${portfolio['total_allocated']:,.0f}")
            
            with col3:
                st.metric("现金储备", f"${portfolio['cash_reserve']:,.0f}")
            
            with col4:
                utilization = portfolio['total_allocated'] / portfolio['total_capital']
                st.metric("资金利用率", f"{utilization:.1%}")
            
            # 投资组合分配表
            if portfolio['allocations']:
                st.subheader("资金分配详情")
                
                allocation_data = []
                for alloc in portfolio['allocations']:
                    allocation_data.append({
                        '股票代码': alloc['symbol'],
                        '股数': alloc['shares'],
                        '价格': f"${alloc['price']:.2f}",
                        '投资金额': f"${alloc['amount']:,.0f}",
                        '仓位比例': f"{alloc['percentage']:.1%}",
                        '预期收益': f"{alloc['expected_return']:.2%}",
                        '操作': alloc['action']
                    })
                
                alloc_df = pd.DataFrame(allocation_data)
                st.dataframe(alloc_df, use_container_width=True)
                
                # 仓位分布饼图
                fig_allocation = go.Figure(data=[go.Pie(
                    labels=[alloc['symbol'] for alloc in portfolio['allocations']],
                    values=[alloc['amount'] for alloc in portfolio['allocations']],
                    title="投资组合分布"
                )])
                
                st.plotly_chart(fig_allocation, use_container_width=True)
            
            # 组合建议
            if portfolio['recommendations']:
                st.subheader("组合级建议")
                for rec in portfolio['recommendations']:
                    st.info(f"💡 {rec}")
        
        # 2. 个股投资建议
        st.subheader("📊 个股投资建议")
        
        # 让用户选择风险偏好来查看建议
        risk_preference = st.selectbox(
            "选择风险偏好查看对应建议：",
            options=RISK_LEVELS,
            index=1  # 默认选择平衡
        )
        
        # 创建建议汇总表
        advice_data = []
        for symbol, strategy in strategies.items():
            if not strategy.get('error', False):
                advice = strategy['recommendations'].get(risk_preference, {})
                
                advice_data.append({
                    '股票代码': symbol,
                    '综合评分': f"{strategy['overall_score']:.3f}",
                    '投资建议': advice.get('action', 'N/A'),
                    '建议仓位': f"{advice.get('position_size', 0):.1%}",
                    '信心等级': advice.get('confidence_level', 'N/A'),
                    '主要理由': advice['reasoning'][0] if advice.get('reasoning') else 'N/A'
                })
        
        if advice_data:
            advice_df = pd.DataFrame(advice_data)
            st.dataframe(advice_df, use_container_width=True)
            
            # 3. 详细建议查看
            st.subheader("📋 详细投资建议")
            
            selected_symbol = st.selectbox(
                "选择股票查看详细建议：",
                options=[item['股票代码'] for item in advice_data]
            )
            
            if selected_symbol:
                strategy = strategies[selected_symbol]
                advice = strategy['recommendations'][risk_preference]
                
                # 显示详细建议
                col1, col2 = st.columns(2)
                
                with col1:
                    st.info(f"**投资行动**: {advice['action']}")
                    st.info(f"**建议仓位**: {advice['position_size']:.1%}")
                    st.info(f"**信心等级**: {advice['confidence_level']}")
                
                with col2:
                    st.info(f"**综合评分**: {strategy['overall_score']:.3f}")
                    
                    if 'current_price' in strategy:
                        current_price = strategy['current_price']
                        predicted_price = strategy['predicted_price']
                        expected_return = (predicted_price - current_price) / current_price
                        st.info(f"**预期收益**: {expected_return:.2%}")
                
                # 决策理由
                st.subheader("决策理由")
                for reason in advice.get('reasoning', []):
                    st.write(f"• {reason}")
                
                # 风险提示
                if advice.get('risk_warnings'):
                    st.subheader("⚠️ 风险提示")
                    for warning in advice['risk_warnings']:
                        st.warning(f"⚠️ {warning}")
                
                # 操作建议
                if advice.get('suggested_actions'):
                    st.subheader("📝 操作建议")
                    for action in advice['suggested_actions']:
                        st.write(f"• {action}")
                
                # 分析详情（展开显示）
                with st.expander("🔍 详细分析数据"):
                    if 'analysis_summary' in strategy:
                        analysis = strategy['analysis_summary']
                        
                        # 情感分析详情
                        st.subheader("情感分析")
                        sentiment = analysis.get('sentiment', {})
                        st.json(sentiment)
                        
                        # 技术分析详情
                        st.subheader("技术分析")
                        technical = analysis.get('technical', {})
                        st.json(technical)
                        
                        # 风险分析详情
                        st.subheader("风险分析")
                        risk = analysis.get('risk', {})
                        st.json(risk)
        
        else:
            st.warning("所有股票的策略生成都失败了")
            
            # 显示错误信息
            st.subheader("错误详情")
            for symbol, strategy in strategies.items():
                if strategy.get('error', False):
                    st.error(f"{symbol}: {strategy.get('error_message', '未知错误')}")
        
        # 4. 风险提示
        st.markdown("---")
        st.error("""
        ⚠️ **重要风险提示**
        
        1. 本系统提供的分析和建议仅供参考，不构成投资建议
        2. 股票投资有风险，过往表现不代表未来收益
        3. AI预测存在不确定性，请结合个人判断谨慎决策
        4. 建议分散投资，控制单一标的仓位
        5. 请根据自身风险承受能力合理配置资产
        
        **投资有风险，入市需谨慎！**
        """)
    
    def _reset_analysis(self):
        """
        重置分析结果
        
        清除所有缓存的分析数据
        """
        st.session_state.market_data = {}
        st.session_state.sentiment_data = {}
        st.session_state.predictions = {}
        st.session_state.strategies = {}
        st.session_state.portfolio = {}
        st.session_state.analysis_completed = False
        st.session_state.last_analysis_time = None
        st.session_state.error_messages = []
        
        st.success("✅ 分析数据已重置")

# 这个文件不包含if __name__ == "__main__"测试代码
# 因为它是Streamlit应用的界面模块，需要通过main.py启动