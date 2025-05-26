# strategy_generator.py
"""
投资策略生成模块

这个模块的作用：
1. 基于AI预测结果生成具体的投资建议
2. 考虑不同投资者的风险偏好
3. 综合技术指标、情感分析、市场环境做出决策
4. 提供仓位管理和风险控制建议

什么是投资策略？
- 将分析结果转化为具体的买卖行动
- 不同的风险偏好对应不同的投资策略
- 需要考虑仓位分配、止盈止损等风险管理
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 导入配置
from config import (
    RISK_LEVELS,      # 风险级别列表
    RISK_PARAMS,      # 风险参数配置
    TRADING_DAYS_PER_YEAR  # 年交易日数
)

class InvestmentStrategyGenerator:
    """
    投资策略生成器类
    
    根据预测结果和风险偏好生成个性化投资建议
    """
    
    def __init__(self):
        """
        初始化策略生成器
        """
        print("正在初始化投资策略生成器...")
        
        # 从配置文件加载风险参数
        self.risk_levels = RISK_LEVELS           # ['保守', '平衡', '激进']
        self.risk_params = RISK_PARAMS           # 各风险级别的参数
        
        # 策略评分权重配置
        # 这些权重决定了各因素对最终决策的影响程度
        self.scoring_weights = {
            'price_prediction': 0.4,    # 价格预测权重40%
            'sentiment': 0.25,          # 情感分析权重25%
            'technical': 0.25,          # 技术指标权重25%
            'risk': 0.1                 # 风险因子权重10%
        }
        
        # 技术指标阈值配置
        self.technical_thresholds = {
            'rsi_oversold': 30,         # RSI超卖线
            'rsi_overbought': 70,       # RSI超买线
            'volatility_high': 0.03,    # 高波动率阈值
            'volume_spike': 1.5         # 成交量放大倍数
        }
        
        print("✅ 策略生成器初始化完成")
        print(f"   支持的风险级别: {self.risk_levels}")
        print(f"   评分权重: {self.scoring_weights}")
    
    def generate_strategy(self, predictions, sentiment_analysis, market_data):
        """
        生成完整的投资策略
        
        这是主要的策略生成方法，整合所有分析结果
        
        参数：
        predictions (dict): AI价格预测结果
        sentiment_analysis (dict): 情感分析结果  
        market_data (dict): 市场技术数据
        
        返回：
        dict: 包含所有股票投资策略的字典
        """
        
        print("正在生成投资策略...")
        print(f"处理 {len(predictions)} 只股票的策略")
        
        strategies = {}  # 存储所有股票的策略
        
        # 遍历每只股票生成策略
        for symbol in predictions.keys():
            try:
                print(f"\n--- 分析 {symbol} ---")
                
                # 获取该股票的各项数据
                prediction_data = predictions[symbol]
                sentiment_data = sentiment_analysis[symbol]
                stock_market_data = market_data[symbol]
                
                # 提取关键数据
                current_price = stock_market_data['price_data']['Close'].iloc[-1]
                predicted_price = prediction_data['predicted_price']
                prediction_confidence = prediction_data['confidence']
                
                print(f"当前价格: ${current_price:.2f}")
                print(f"预测价格: ${predicted_price:.2f}")
                print(f"预测置信度: {prediction_confidence:.3f}")
                
                # 计算预期收益率
                expected_return = (predicted_price - current_price) / current_price
                print(f"预期收益率: {expected_return:.3%}")
                
                # 分析技术指标
                technical_analysis = self._analyze_technical_indicators(
                    stock_market_data['price_data']
                )
                
                # 分析风险因子
                risk_analysis = self._analyze_risk_factors(
                    stock_market_data['price_data'],
                    sentiment_data
                )
                
                # 计算综合评分
                overall_score = self._calculate_overall_score(
                    expected_return=expected_return,
                    confidence=prediction_confidence,
                    sentiment=sentiment_data,
                    technical=technical_analysis,
                    risk=risk_analysis
                )
                
                # 为每种风险偏好生成具体建议
                strategy = self._create_investment_advice(
                    symbol=symbol,
                    current_price=current_price,
                    predicted_price=predicted_price,
                    expected_return=expected_return,
                    confidence=prediction_confidence,
                    sentiment=sentiment_data,
                    technical=technical_analysis,
                    risk=risk_analysis,
                    overall_score=overall_score
                )
                
                strategies[symbol] = strategy
                print(f"✅ {symbol} 策略生成完成")
                
            except Exception as e:
                print(f"❌ {symbol} 策略生成失败: {e}")
                # 创建一个失败的策略记录
                strategies[symbol] = self._create_error_strategy(symbol, str(e))
        
        print(f"\n✅ 投资策略生成完成，共处理 {len(strategies)} 只股票")
        return strategies
    
    def _analyze_technical_indicators(self, price_data):
        """
        分析技术指标
        
        从技术分析角度评估股票的买卖信号
        
        参数：
        price_data (DataFrame): 股票价格数据（包含技术指标）
        
        返回：
        dict: 技术指标分析结果
        """
        
        print("  分析技术指标...")
        
        # 获取最新的技术指标值
        latest_data = price_data.iloc[-1]
        
        analysis = {
            'signals': [],          # 技术信号列表
            'strength': 0,          # 技术面强度 (-1 到 1)
            'details': {}           # 详细指标值
        }
        
        # 1. RSI分析
        if 'RSI' in price_data.columns:
            rsi = latest_data['RSI']
            analysis['details']['RSI'] = rsi
            
            if rsi <= self.technical_thresholds['rsi_oversold']:
                analysis['signals'].append('RSI超卖_买入信号')
                analysis['strength'] += 0.3
            elif rsi >= self.technical_thresholds['rsi_overbought']:
                analysis['signals'].append('RSI超买_卖出信号')
                analysis['strength'] -= 0.3
            else:
                analysis['signals'].append('RSI正常区间')
        
        # 2. 移动平均线分析
        if all(col in price_data.columns for col in ['Close', 'MA20', 'MA50']):
            close = latest_data['Close']
            ma20 = latest_data['MA20']
            ma50 = latest_data['MA50']
            
            analysis['details']['MA20'] = ma20
            analysis['details']['MA50'] = ma50
            
            # 价格与均线关系
            if close > ma20 > ma50:
                analysis['signals'].append('价格站上均线_上涨趋势')
                analysis['strength'] += 0.2
            elif close < ma20 < ma50:
                analysis['signals'].append('价格跌破均线_下跌趋势')
                analysis['strength'] -= 0.2
            else:
                analysis['signals'].append('均线纠缠_震荡行情')
        
        # 3. MACD分析
        if all(col in price_data.columns for col in ['MACD', 'MACD_Signal']):
            macd = latest_data['MACD']
            macd_signal = latest_data['MACD_Signal']
            
            analysis['details']['MACD'] = macd
            analysis['details']['MACD_Signal'] = macd_signal
            
            if macd > macd_signal and macd > 0:
                analysis['signals'].append('MACD金叉_买入信号')
                analysis['strength'] += 0.25
            elif macd < macd_signal and macd < 0:
                analysis['signals'].append('MACD死叉_卖出信号')
                analysis['strength'] -= 0.25
        
        # 4. 成交量分析
        if all(col in price_data.columns for col in ['Volume', 'Volume_MA']):
            volume = latest_data['Volume']
            volume_ma = latest_data['Volume_MA']
            
            analysis['details']['Volume'] = volume
            analysis['details']['Volume_MA'] = volume_ma
            
            if volume > volume_ma * self.technical_thresholds['volume_spike']:
                analysis['signals'].append('成交量放大_关注度提升')
                analysis['strength'] += 0.1
            elif volume < volume_ma * 0.5:
                analysis['signals'].append('成交量萎缩_关注度下降')
                analysis['strength'] -= 0.05
        
        # 5. 波动率分析
        if 'Volatility' in price_data.columns:
            volatility = latest_data['Volatility']
            analysis['details']['Volatility'] = volatility
            
            if volatility > self.technical_thresholds['volatility_high']:
                analysis['signals'].append('高波动率_风险加大')
                analysis['strength'] -= 0.1
        
        # 限制强度范围在 -1 到 1 之间
        analysis['strength'] = max(-1, min(1, analysis['strength']))
        
        print(f"    技术面强度: {analysis['strength']:.2f}")
        print(f"    主要信号: {', '.join(analysis['signals'][:3])}")
        
        return analysis
    
    def _analyze_risk_factors(self, price_data, sentiment_data):
        """
        分析风险因子
        
        评估投资该股票的风险水平
        
        参数：
        price_data (DataFrame): 价格数据
        sentiment_data (dict): 情感数据
        
        返回：
        dict: 风险分析结果
        """
        
        print("  分析风险因子...")
        
        risk_analysis = {
            'overall_risk': 0,      # 总体风险水平 (0-1)
            'risk_factors': [],     # 风险因子列表
            'details': {}           # 详细风险指标
        }
        
        # 1. 价格波动风险
        if 'Volatility' in price_data.columns:
            volatility = price_data['Volatility'].iloc[-1]
            volatility_percentile = self._calculate_percentile(
                price_data['Volatility'].dropna(), volatility
            )
            
            risk_analysis['details']['volatility'] = volatility
            risk_analysis['details']['volatility_percentile'] = volatility_percentile
            
            if volatility_percentile > 0.8:
                risk_analysis['risk_factors'].append('极高波动率')
                risk_analysis['overall_risk'] += 0.3
            elif volatility_percentile > 0.6:
                risk_analysis['risk_factors'].append('高波动率')
                risk_analysis['overall_risk'] += 0.2
        
        # 2. 价格变化风险
        if 'Price_Change' in price_data.columns:
            recent_changes = price_data['Price_Change'].tail(5).abs()
            avg_change = recent_changes.mean()
            
            risk_analysis['details']['recent_volatility'] = avg_change
            
            if avg_change > 0.05:  # 日均变化超过5%
                risk_analysis['risk_factors'].append('价格剧烈波动')
                risk_analysis['overall_risk'] += 0.2
        
        # 3. 情感风险
        sentiment_confidence = sentiment_data.get('confidence', 0)
        news_count = sentiment_data.get('news_count', 0)
        
        risk_analysis['details']['sentiment_confidence'] = sentiment_confidence
        risk_analysis['details']['news_count'] = news_count
        
        if sentiment_confidence < 0.3:
            risk_analysis['risk_factors'].append('情感分析不确定')
            risk_analysis['overall_risk'] += 0.15
        
        if news_count < 3:
            risk_analysis['risk_factors'].append('新闻关注度低')
            risk_analysis['overall_risk'] += 0.1
        
        # 4. 趋势一致性风险
        if all(col in price_data.columns for col in ['MA5', 'MA20', 'MA50']):
            latest = price_data.iloc[-1]
            ma5, ma20, ma50 = latest['MA5'], latest['MA20'], latest['MA50']
            
            # 检查均线排列是否一致
            if not ((ma5 > ma20 > ma50) or (ma5 < ma20 < ma50)):
                risk_analysis['risk_factors'].append('趋势不明确')
                risk_analysis['overall_risk'] += 0.1
        
        # 限制总体风险在 0-1 范围内
        risk_analysis['overall_risk'] = min(1, risk_analysis['overall_risk'])
        
        print(f"    总体风险水平: {risk_analysis['overall_risk']:.2f}")
        if risk_analysis['risk_factors']:
            print(f"    主要风险: {', '.join(risk_analysis['risk_factors'][:2])}")
        
        return risk_analysis
    
    def _calculate_overall_score(self, expected_return, confidence, sentiment, technical, risk):
        """
        计算综合投资评分
        
        整合所有分析维度，得出一个综合评分
        
        参数：
        expected_return (float): 预期收益率
        confidence (float): 预测置信度
        sentiment (dict): 情感分析结果
        technical (dict): 技术分析结果
        risk (dict): 风险分析结果
        
        返回：
        dict: 综合评分结果
        """
        
        print("  计算综合评分...")
        
        # 1. 价格预测评分 (-1 到 1)
        # 结合预期收益率和置信度
        price_score = expected_return * confidence
        price_score = max(-1, min(1, price_score))  # 限制范围
        
        # 2. 情感评分 (-1 到 1)
        sentiment_score = sentiment.get('overall_sentiment', 0)
        sentiment_confidence = sentiment.get('confidence', 0)
        # 用置信度调整情感得分的权重
        adjusted_sentiment_score = sentiment_score * sentiment_confidence
        
        # 3. 技术评分 (-1 到 1)
        technical_score = technical.get('strength', 0)
        
        # 4. 风险评分 (0 到 -1，风险越高分数越低)
        risk_score = -risk.get('overall_risk', 0)
        
        # 5. 计算加权综合评分
        overall_score = (
            price_score * self.scoring_weights['price_prediction'] +
            adjusted_sentiment_score * self.scoring_weights['sentiment'] +
            technical_score * self.scoring_weights['technical'] +
            risk_score * self.scoring_weights['risk']
        )
        
        # 确保最终评分在 -1 到 1 范围内
        overall_score = max(-1, min(1, overall_score))
        
        print(f"    价格预测评分: {price_score:.3f}")
        print(f"    情感评分: {adjusted_sentiment_score:.3f}")
        print(f"    技术评分: {technical_score:.3f}")
        print(f"    风险评分: {risk_score:.3f}")
        print(f"    综合评分: {overall_score:.3f}")
        
        return {
            'overall_score': overall_score,
            'component_scores': {
                'price_prediction': price_score,
                'sentiment': adjusted_sentiment_score,
                'technical': technical_score,
                'risk': risk_score
            },
            'weights': self.scoring_weights.copy()
        }
    
    def _create_investment_advice(self, symbol, current_price, predicted_price, 
                                expected_return, confidence, sentiment, technical, 
                                risk, overall_score):
        """
        创建具体的投资建议
        
        根据综合分析为不同风险偏好的投资者生成建议
        
        参数：
        symbol (str): 股票代码
        current_price (float): 当前价格
        predicted_price (float): 预测价格
        expected_return (float): 预期收益率
        confidence (float): 预测置信度
        sentiment (dict): 情感分析
        technical (dict): 技术分析
        risk (dict): 风险分析
        overall_score (dict): 综合评分
        
        返回：
        dict: 完整的投资建议
        """
        
        print("  生成投资建议...")
        
        # 基础信息
        advice = {
            'symbol': symbol,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'expected_return': expected_return,
            'confidence': confidence,
            'overall_score': overall_score['overall_score'],
            'analysis_summary': {
                'sentiment': sentiment,
                'technical': technical,
                'risk': risk,
                'scoring': overall_score
            },
            'recommendations': {},  # 存储不同风险偏好的建议
            'generated_at': datetime.now().isoformat()
        }
        
        # 为每种风险偏好生成建议
        for risk_level in self.risk_levels:
            recommendation = self._generate_risk_specific_advice(
                risk_level, overall_score['overall_score'], 
                expected_return, confidence, risk['overall_risk']
            )
            advice['recommendations'][risk_level] = recommendation
        
        return advice
    
    def _generate_risk_specific_advice(self, risk_level, overall_score, 
                                     expected_return, confidence, risk_level_score):
        """
        为特定风险偏好生成建议
        
        参数：
        risk_level (str): 风险偏好级别
        overall_score (float): 综合评分
        expected_return (float): 预期收益率
        confidence (float): 置信度
        risk_level_score (float): 风险水平
        
        返回：
        dict: 风险特定的投资建议
        """
        
        # 获取该风险级别的参数
        params = self.risk_params[risk_level]
        threshold = params['threshold']              # 收益率阈值
        max_position = params['max_position']        # 最大仓位
        confidence_min = params['confidence_min']    # 最小置信度要求
        
        # 初始化建议
        recommendation = {
            'action': '持有',           # 默认动作
            'position_size': 0,        # 建议仓位大小
            'confidence_level': '低',   # 建议置信度
            'reasoning': [],           # 决策理由
            'risk_warnings': [],       # 风险提示
            'suggested_actions': []    # 具体行动建议
        }
        
        # 决策逻辑
        print(f"    为{risk_level}投资者生成建议...")
        
        # 1. 检查基本条件
        if confidence < confidence_min:
            recommendation['reasoning'].append(f'预测置信度({confidence:.2f})低于{risk_level}投资者要求({confidence_min})')
            recommendation['action'] = '观望'
            return recommendation
        
        # 2. 基于综合评分和预期收益率决定动作
        if overall_score > 0.3 and expected_return > threshold:
            # 强烈买入信号
            recommendation['action'] = '买入'
            base_position = min(max_position, overall_score * max_position)
            
            # 根据置信度调整仓位
            confidence_adjustment = confidence ** 0.5  # 开根号降低置信度影响
            recommendation['position_size'] = base_position * confidence_adjustment
            
            recommendation['confidence_level'] = '高' if overall_score > 0.6 else '中'
            recommendation['reasoning'].append(f'综合评分({overall_score:.2f})积极，预期收益({expected_return:.2%})超过阈值')
            
        elif overall_score < -0.3 and expected_return < -threshold:
            # 强烈卖出信号
            recommendation['action'] = '卖出'
            recommendation['position_size'] = min(max_position, abs(overall_score) * max_position)
            recommendation['confidence_level'] = '高' if overall_score < -0.6 else '中'
            recommendation['reasoning'].append(f'综合评分({overall_score:.2f})消极，预期下跌({expected_return:.2%})')
            
        elif overall_score > 0.1 and expected_return > 0:
            # 轻度买入信号
            recommendation['action'] = '小额买入'
            recommendation['position_size'] = max_position * 0.3 * overall_score
            recommendation['confidence_level'] = '中'
            recommendation['reasoning'].append('轻度积极信号，建议小额试探')
            
        elif overall_score < -0.1 and expected_return < 0:
            # 轻度卖出信号
            recommendation['action'] = '减仓'
            recommendation['position_size'] = max_position * 0.3 * abs(overall_score)
            recommendation['confidence_level'] = '中'
            recommendation['reasoning'].append('轻度消极信号，建议适当减仓')
            
        else:
            # 中性信号
            recommendation['action'] = '持有'
            recommendation['position_size'] = 0
            recommendation['confidence_level'] = '低'
            recommendation['reasoning'].append('市场信号不明确，建议继续观察')
        
        # 3. 添加风险调整
        if risk_level_score > 0.5:
            # 高风险调整
            recommendation['position_size'] *= 0.7  # 减少仓位
            recommendation['risk_warnings'].append('高风险股票，建议降低仓位')
            
            if risk_level == '保守':
                recommendation['position_size'] *= 0.5  # 保守投资者进一步降低
                recommendation['risk_warnings'].append('不适合保守投资者的风险水平')
        
        # 4. 生成具体行动建议
        if recommendation['action'] in ['买入', '小额买入']:
            recommendation['suggested_actions'].extend([
                '分批建仓，避免一次性投入',
                '设置止损位，控制下行风险',
                '关注技术指标变化，及时调整'
            ])
        elif recommendation['action'] in ['卖出', '减仓']:
            recommendation['suggested_actions'].extend([
                '分批卖出，避免冲击成本',
                '等待反弹机会减少损失',
                '密切关注市场情绪变化'
            ])
        else:  # 持有或观望
            recommendation['suggested_actions'].extend([
                '继续观察价格走势',
                '关注重要技术位突破',
                '留意基本面变化信号'
            ])
        
        # 5. 限制仓位范围
        recommendation['position_size'] = max(0, min(max_position, recommendation['position_size']))
        
        print(f"      {risk_level}: {recommendation['action']} "
              f"(仓位: {recommendation['position_size']:.1%})")
        
        return recommendation
    
    def _calculate_percentile(self, data_series, value):
        """
        计算数值在序列中的百分位数
        
        参数：
        data_series (pd.Series): 数据序列
        value (float): 要计算百分位的数值
        
        返回：
        float: 百分位数 (0-1)
        """
        return (data_series <= value).mean()
    
    def _create_error_strategy(self, symbol, error_message):
        """
        创建错误处理策略
        
        当某只股票分析失败时返回的策略
        
        参数：
        symbol (str): 股票代码
        error_message (str): 错误信息
        
        返回：
        dict: 错误策略
        """
        return {
            'symbol': symbol,
            'error': True,
            'error_message': error_message,
            'recommendations': {
                risk_level: {
                    'action': '暂停分析',
                    'position_size': 0,
                    'confidence_level': '无',
                    'reasoning': [f'分析失败: {error_message}'],
                    'risk_warnings': ['数据分析异常，建议人工核查'],
                    'suggested_actions': ['等待数据修复后重新分析']
                }
                for risk_level in self.risk_levels
            },
            'generated_at': datetime.now().isoformat()
        }
    
    def generate_portfolio_advice(self, strategies, risk_preference, total_capital=100000):
        """
        生成投资组合建议
        
        基于单只股票的策略生成整体投资组合配置
        
        参数：
        strategies (dict): 所有股票的策略
        risk_preference (str): 投资者风险偏好
        total_capital (float): 总投资资金
        
        返回：
        dict: 投资组合建议
        """
        
        print(f"\n正在生成{risk_preference}投资组合建议...")
        print(f"总资金: ${total_capital:,.2f}")
        
        portfolio = {
            'total_capital': total_capital,
            'risk_preference': risk_preference,
            'allocations': [],          # 资金分配列表
            'total_allocated': 0,       # 总分配资金
            'cash_reserve': 0,          # 现金储备
            'expected_return': 0,       # 预期组合收益
            'risk_level': 0,           # 组合风险水平
            'recommendations': []       # 组合级别建议
        }
        
        # 收集有效策略
        valid_strategies = []
        for symbol, strategy in strategies.items():
            if not strategy.get('error', False):
                risk_advice = strategy['recommendations'].get(risk_preference)
                if risk_advice and risk_advice['action'] in ['买入', '小额买入']:
                    valid_strategies.append((symbol, strategy, risk_advice))
        
        if not valid_strategies:
            portfolio['recommendations'].append('当前市场环境下无合适投资标的，建议持币观望')
            portfolio['cash_reserve'] = total_capital
            return portfolio
        
        # 按综合评分排序
        valid_strategies.sort(key=lambda x: x[1]['overall_score'], reverse=True)
        
        # 分配资金
        allocated_capital = 0
        max_positions = min(len(valid_strategies), 5)  # 最多投资5只股票
        
        for i, (symbol, strategy, risk_advice) in enumerate(valid_strategies[:max_positions]):
            # 计算分配金额
            position_ratio = risk_advice['position_size']
            
            # 根据排名调整权重（第一名权重最高）
            rank_weight = 1.0 - (i * 0.1)  # 每降一位减少10%
            adjusted_ratio = position_ratio * rank_weight
            
            allocation_amount = total_capital * adjusted_ratio
            
            # 确保不超过剩余资金
            remaining_capital = total_capital - allocated_capital
            allocation_amount = min(allocation_amount, remaining_capital * 0.8)  # 最多用80%资金
            
            if allocation_amount >= 1000:  # 最小投资门槛
                shares = int(allocation_amount // strategy['current_price'])
                actual_amount = shares * strategy['current_price']
                
                portfolio['allocations'].append({
                    'symbol': symbol,
                    'shares': shares,
                    'price': strategy['current_price'],
                    'amount': actual_amount,
                    'percentage': actual_amount / total_capital,
                    'expected_return': strategy['expected_return'],
                    'action': risk_advice['action'],
                    'reasoning': risk_advice['reasoning'][0] if risk_advice['reasoning'] else ''
                })
                
                allocated_capital += actual_amount
                
                # 更新组合指标
                weight = actual_amount / total_capital
                portfolio['expected_return'] += strategy['expected_return'] * weight
                portfolio['risk_level'] += strategy['analysis_summary']['risk']['overall_risk'] * weight
        
        portfolio['total_allocated'] = allocated_capital
        portfolio['cash_reserve'] = total_capital - allocated_capital
        
        # 生成组合级建议
        if len(portfolio['allocations']) > 0:
            portfolio['recommendations'].extend([
                f'建议投资 {len(portfolio['allocations'])} 只股票',
                f'资金利用率: {allocated_capital/total_capital:.1%}',
                f'预期组合收益: {portfolio["expected_return"]:.2%}',
                f'组合风险水平: {portfolio["risk_level"]:.2f}'
            ])
        
        print(f"✅ 投资组合生成完成")
        print(f"   投资标的: {len(portfolio['allocations'])} 只")
        print(f"   资金利用率: {allocated_capital/total_capital:.1%}")
        
        return portfolio

# 模块测试代码
if __name__ == "__main__":
    print("=== 投资策略生成模块测试 ===")
    
    # 创建策略生成器
    generator = InvestmentStrategyGenerator()
    
    # 创建模拟数据
    print("\n--- 创建模拟数据 ---")
    
    mock_predictions = {
        'AAPL': {
            'predicted_price': 155.0,
            'confidence': 0.8
        },
        'MSFT': {
            'predicted_price': 320.0,
            'confidence': 0.75
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
    import pandas as pd
    import numpy as np
    
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    mock_market_data = {}
    for symbol in ['AAPL', 'MSFT']:
        prices = 150 + np.cumsum(np.random.randn(100) * 0.02)
        
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
    print("\n--- 测试策略生成 ---")
    try:
        strategies = generator.generate_strategy(
            mock_predictions, mock_sentiment, mock_market_data
        )
        
        print("✅ 策略生成成功")
        for symbol, strategy in strategies.items():
            print(f"{symbol}: {strategy['overall_score']:.3f}")
            for risk_level in generator.risk_levels:
                advice = strategy['recommendations'][risk_level]
                print(f"  {risk_level}: {advice['action']} ({advice['position_size']:.1%})")
    
    except Exception as e:
        print(f"❌ 策略生成失败: {e}")
    
    # 测试投资组合生成
    print("\n--- 测试投资组合生成 ---")
    try:
        portfolio = generator.generate_portfolio_advice(strategies, '平衡', 100000)
        print("✅ 投资组合生成成功")
        print(f"投资标的数量: {len(portfolio['allocations'])}")
        print(f"资金利用率: {portfolio['total_allocated']/portfolio['total_capital']:.1%}")
        
    except Exception as e:
        print(f"❌ 投资组合生成失败: {e}")
    
    print("\n=== 测试完成 ===")