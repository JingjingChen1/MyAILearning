# price_predictor.py
"""
AI价格预测模块

这个模块的作用：
1. 使用LSTM神经网络预测股票价格
2. 结合技术指标、情感分析、经济数据进行特征工程
3. 训练深度学习模型学习股价变化规律
4. 提供未来价格预测和置信度评估

什么是LSTM？
- Long Short-Term Memory（长短期记忆网络）
- 一种特殊的循环神经网络(RNN)，擅长处理时间序列数据
- 能够记住长期依赖关系，适合预测股价这种有时间相关性的数据
- 比传统方法更能捕捉复杂的价格变化模式
"""

# 导入深度学习和机器学习相关库
import tensorflow as tf          # Google的深度学习框架
from sklearn.preprocessing import StandardScaler  # 数据标准化工具
from sklearn.model_selection import train_test_split  # 数据集分割工具
from sklearn.metrics import mean_squared_error, mean_absolute_error  # 评估指标
import joblib                   # 模型保存和加载工具

# 导入数据处理库
import pandas as pd             # 数据处理
import numpy as np              # 数学计算
import warnings                # 警告管理

# 导入配置
from config import LSTM_CONFIG, FEATURE_CONFIG, ERROR_MESSAGES

# 忽略TensorFlow的一些警告信息
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 减少TensorFlow日志输出

class StockPricePredictor:
    """
    股价预测器类
    
    使用LSTM神经网络进行股价预测
    包含完整的数据预处理、模型训练、预测流程
    """
    
    def __init__(self):
        """
        初始化预测器
        
        设置模型参数和工具
        """
        print("正在初始化股价预测器...")
        
        # 数据标准化工具
        # StandardScaler用于将数据标准化到均值0、标准差1的分布
        # 为什么要标准化？神经网络对输入数据的尺度很敏感
        self.scaler = StandardScaler()
        
        # 模型相关属性
        self.model = None                                    # LSTM模型，初始为空
        self.sequence_length = LSTM_CONFIG['sequence_length'] # 时间序列长度(30天)
        self.is_trained = False                              # 模型是否已训练
        self.training_history = None                         # 训练历史记录
        
        # 特征列名（训练后保存，预测时验证）
        self.feature_columns = None
        
        # 从配置文件获取模型参数
        self.epochs = LSTM_CONFIG['epochs']                         # 训练轮数
        self.batch_size = LSTM_CONFIG['batch_size']                 # 批次大小
        self.learning_rate = LSTM_CONFIG['learning_rate']           # 学习率
        self.validation_split = LSTM_CONFIG['validation_split']     # 验证集比例
        
        print(f"✅ 预测器初始化完成")
        print(f"   序列长度: {self.sequence_length}")
        print(f"   训练轮数: {self.epochs}")
        print(f"   批次大小: {self.batch_size}")
    
    def prepare_features(self, stock_data, sentiment_data, economic_data):
        """
        准备训练特征
        
        特征工程是机器学习中最重要的步骤之一：
        - 选择对预测有用的特征
        - 组合多种数据源（股价+情感+经济指标）
        - 处理缺失值和异常值
        
        参数：
        stock_data (dict): 股票数据，包含price_data等
        sentiment_data (dict): 情感分析数据
        economic_data (dict): 经济指标数据
        
        返回：
        pandas.DataFrame: 整理好的特征数据
        """
        
        print("正在准备训练特征...")
        
        # 复制股票价格数据，避免修改原始数据
        df = stock_data['price_data'].copy()
        
        print(f"  原始数据形状: {df.shape}")
        
        # 1. 基础价格特征
        # 这些是从股价数据直接获得的特征
        price_features = [
            'Close',         # 收盘价（最重要的特征）
            'Open',          # 开盘价
            'High',          # 最高价
            'Low',           # 最低价
            'Volume'         # 成交量
        ]
        
        # 2. 技术指标特征
        # 这些是从价格数据计算出的技术分析指标
        technical_features = [
            'MA5', 'MA20', 'MA50',          # 移动平均线
            'RSI',                          # 相对强弱指标
            'MACD', 'MACD_Signal',          # MACD指标
            'Volatility',                   # 波动率
            'Price_Change'                  # 价格变化率
        ]
        
        # 检查技术指标是否存在
        available_technical = [col for col in technical_features if col in df.columns]
        if len(available_technical) < len(technical_features):
            missing = set(technical_features) - set(available_technical)
            print(f"  警告：缺少技术指标 {missing}")
        
        # 3. 添加价格关系特征
        # 这些特征帮助模型理解价格之间的关系
        print("  计算价格关系特征...")
        
        # 价格位置特征：收盘价在高低价区间中的位置
        df['Price_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-8)
        
        # 价格与移动平均线的关系
        if 'MA20' in df.columns:
            df['Price_MA20_Ratio'] = df['Close'] / df['MA20']  # 价格相对MA20的比率
        
        if 'MA50' in df.columns:
            df['Price_MA50_Ratio'] = df['Close'] / df['MA50']  # 价格相对MA50的比率
        
        # 成交量相对特征
        if 'Volume_MA' in df.columns:
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']  # 成交量相对平均的比率
        else:
            # 如果没有Volume_MA，计算20日成交量均值
            df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # 4. 添加时间特征
        # 股市有明显的时间规律（周一效应、月末效应等）
        print("  添加时间特征...")
        
        # 确保索引是DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            print("    警告：索引不是日期格式，跳过时间特征")
        else:
            df['Day_of_Week'] = df.index.dayofweek      # 星期几 (0=周一)
            df['Month'] = df.index.month                # 月份
            df['Quarter'] = df.index.quarter            # 季度
        
        # 5. 添加情感分析特征
        print("  添加情感分析特征...")
        
        # 情感数据通常是针对最近时期的，这里简化处理
        # 给所有日期分配相同的情感得分
        sentiment_score = sentiment_data.get('overall_sentiment', 0.0)
        sentiment_confidence = sentiment_data.get('confidence', 0.0)
        
        df['Sentiment_Score'] = sentiment_score        # 情感得分
        df['Sentiment_Confidence'] = sentiment_confidence  # 情感置信度
        
        # 情感强度（绝对值）
        df['Sentiment_Strength'] = abs(sentiment_score)
        
        # 6. 添加宏观经济特征
        print("  添加宏观经济特征...")
        
        # 处理经济指标，转换为综合趋势分数
        economic_trend = self._process_economic_indicators(economic_data)
        df['Economic_Trend'] = economic_trend
        
        # 7. 添加滞后特征
        # 滞后特征能帮助模型学习时间依赖关系
        print("  添加滞后特征...")
        
        lag_periods = [1, 2, 3, 5]  # 1天、2天、3天、5天前的数据
        for lag in lag_periods:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
        
        # 8. 添加移动统计特征
        print("  添加移动统计特征...")
        
        # 5日收益率
        df['Return_5d'] = df['Close'].pct_change(5)
        
        # 10日最高价和最低价
        df['High_10d'] = df['High'].rolling(window=10).max()
        df['Low_10d'] = df['Low'].rolling(window=10).min()
        
        # 价格在N日区间中的位置
        df['Price_Percentile_10d'] = (df['Close'] - df['Low_10d']) / (df['High_10d'] - df['Low_10d'] + 1e-8)
        
        # 9. 确定最终特征列表
        feature_columns = []
        
        # 添加基础价格特征
        feature_columns.extend([col for col in price_features if col in df.columns])
        
        # 添加技术指标特征
        feature_columns.extend(available_technical)
        
        # 添加计算出的特征
        computed_features = [
            'Price_Position', 'Price_MA20_Ratio', 'Price_MA50_Ratio',
            'Volume_Ratio', 'Sentiment_Score', 'Sentiment_Confidence', 
            'Sentiment_Strength', 'Economic_Trend', 'Return_5d',
            'Price_Percentile_10d'
        ]
        
        # 添加时间特征（如果可用）
        if 'Day_of_Week' in df.columns:
            computed_features.extend(['Day_of_Week', 'Month', 'Quarter'])
        
        # 添加滞后特征
        for lag in lag_periods:
            computed_features.extend([f'Close_Lag_{lag}', f'Volume_Lag_{lag}'])
        
        # 只添加实际存在的特征
        for feature in computed_features:
            if feature in df.columns:
                feature_columns.append(feature)
        
        # 10. 数据清理
        print("  清理数据...")
        
        # 移除包含缺失值的行
        initial_rows = len(df)
        df = df.dropna()
        final_rows = len(df)
        
        if final_rows < initial_rows:
            print(f"  移除了 {initial_rows - final_rows} 行包含缺失值的数据")
        
        # 检查数据量是否足够
        min_required = self.sequence_length + 50  # 至少需要序列长度+50个数据点
        if final_rows < min_required:
            raise ValueError(f"数据不足：只有 {final_rows} 行，至少需要 {min_required} 行")
        
        # 保存特征列名，用于后续验证
        self.feature_columns = feature_columns.copy()
        
        print(f"  ✅ 特征准备完成")
        print(f"     最终数据形状: {df[feature_columns].shape}")
        print(f"     特征数量: {len(feature_columns)}")
        print(f"     特征列表: {feature_columns}")
        
        # 返回只包含选定特征的数据
        return df[feature_columns]
    
    def _process_economic_indicators(self, economic_data):
        """
        处理经济指标数据
        
        将多个经济指标综合成一个趋势分数
        简化模型输入，避免特征过多
        
        参数：
        economic_data (dict): 经济指标数据字典
        
        返回：
        float: 综合经济趋势分数
        """
        
        trend_score = 0      # 趋势分数累计
        valid_indicators = 0 # 有效指标数量
        
        # 定义各指标对股市的影响权重
        # 正权重表示该指标上升对股市有利
        # 负权重表示该指标上升对股市不利
        indicator_weights = {
            'GDP': 1.0,           # GDP增长对股市积极
            'Inflation': -0.5,    # 通胀过高对股市不利
            'Unemployment': -1.0, # 失业率高对股市不利
            'Interest_Rate': -0.8 # 利率高对股市不利（资金成本上升）
        }
        
        # 遍历所有经济指标
        for indicator, data_points in economic_data.items():
            try:
                # 确保至少有两个数据点来计算趋势
                if len(data_points) >= 2:
                    # 提取最近两个时期的数值
                    recent_values = []
                    for dp in data_points[-2:]:  # 取最后两个数据点
                        if dp['value'] is not None:
                            recent_values.append(dp['value'])
                    
                    # 需要两个有效数值才能计算变化率
                    if len(recent_values) == 2:
                        # 计算变化率：(新值 - 旧值) / 旧值
                        old_value, new_value = recent_values
                        change_rate = (new_value - old_value) / abs(old_value) if old_value != 0 else 0
                        
                        # 获取该指标的权重
                        weight = indicator_weights.get(indicator, 0.5)
                        
                        # 累加加权变化率
                        trend_score += change_rate * weight
                        valid_indicators += 1
                        
                        print(f"    {indicator}: {old_value:.2f} -> {new_value:.2f}, "
                              f"变化率: {change_rate:.3f}, 权重: {weight}")
                        
            except Exception as e:
                print(f"    处理 {indicator} 指标时出错: {e}")
                continue
        
        # 返回平均趋势分数
        if valid_indicators > 0:
            final_score = trend_score / valid_indicators
        else:
            final_score = 0
        
        print(f"    综合经济趋势分数: {final_score:.3f}")
        return final_score
    
    def create_sequences(self, data, target_column='Close'):
        """
        创建时序训练数据
        
        将时间序列数据转换为LSTM可以使用的格式
        用过去N天的数据预测第N+1天的值
        
        参数：
        data (DataFrame): 特征数据
        target_column (str): 要预测的目标列（默认是收盘价）
        
        返回：
        tuple: (X, y) 其中X是输入序列，y是目标值
        """
        
        print(f"正在创建时序数据，序列长度: {self.sequence_length}")
        
        # 检查目标列是否存在
        if target_column not in data.columns:
            raise ValueError(f"目标列 '{target_column}' 不存在于数据中")
        
        X, y = [], []  # 初始化输入和输出列表
        
        # 从第sequence_length天开始，因为需要前面的数据作为输入
        for i in range(self.sequence_length, len(data)):
            # 输入序列：第i-sequence_length天到第i-1天的所有特征
            # iloc[start:end] 选择从start到end-1的行
            input_sequence = data.iloc[i-self.sequence_length:i].values
            X.append(input_sequence)
            
            # 目标值：第i天的收盘价
            target_value = data[target_column].iloc[i]
            y.append(target_value)
        
        # 转换为numpy数组，这是TensorFlow需要的格式
        X = np.array(X)
        y = np.array(y)
        
        print(f"  ✅ 序列创建完成")
        print(f"     输入形状: {X.shape}")  # (样本数, 时间步, 特征数)
        print(f"     输出形状: {y.shape}")  # (样本数,)
        print(f"     总样本数: {len(X)}")
        
        return X, y
    
    def build_model(self, input_shape):
        """
        构建LSTM预测模型
        
        设计神经网络架构：
        - 多层LSTM：学习不同层次的时间模式
        - Dropout层：防止过拟合
        - Dense层：最终的价格预测
        
        参数：
        input_shape (tuple): 输入数据的形状 (时间步长, 特征数量)
        
        返回：
        tensorflow.keras.Model: 构建好的模型
        """
        
        print(f"正在构建LSTM模型，输入形状: {input_shape}")
        
        # 使用Sequential模型，层按顺序堆叠
        model = tf.keras.Sequential()
        
        # 第一层LSTM
        # 64个神经元，return_sequences=True表示返回完整序列
        model.add(tf.keras.layers.LSTM(
            64,                           # 64个LSTM单元
            return_sequences=True,        # 返回完整序列给下一层
            input_shape=input_shape,      # 输入数据形状
            name='lstm_1'                 # 层的名称
        ))
        
        # 第一个Dropout层：随机关闭20%的神经元，防止过拟合
        model.add(tf.keras.layers.Dropout(0.2, name='dropout_1'))
        
        # 第二层LSTM（神经元数量递减）
        model.add(tf.keras.layers.LSTM(
            32,                           # 32个LSTM单元
            return_sequences=True,        # 继续返回序列
            name='lstm_2'
        ))
        model.add(tf.keras.layers.Dropout(0.2, name='dropout_2'))
        
        # 第三层LSTM（最后一层LSTM）
        model.add(tf.keras.layers.LSTM(
            16,                           # 16个LSTM单元
            return_sequences=False,       # 只返回最后的输出
            name='lstm_3'
        ))
        model.add(tf.keras.layers.Dropout(0.2, name='dropout_3'))
        
        # 全连接层（Dense层）
        # 逐步减少神经元数量，最终输出1个值（预测价格）
        model.add(tf.keras.layers.Dense(32, activation='relu', name='dense_1'))
        model.add(tf.keras.layers.Dropout(0.1, name='dropout_4'))
        
        model.add(tf.keras.layers.Dense(16, activation='relu', name='dense_2'))
        
        # 输出层：1个神经元，无激活函数（线性输出）
        model.add(tf.keras.layers.Dense(1, name='output'))
        
        # 编译模型：设置优化器、损失函数和评估指标
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',                    # 均方误差损失（适合回归问题）
            metrics=['mae']                # 平均绝对误差作为监控指标
        )
        
        # 打印模型结构
        print("✅ 模型构建完成")
        model.summary()  # 显示模型的详细结构
        
        return model
    
    def train_model(self, features_data):
        """
        训练LSTM模型
        
        完整的训练流程：
        1. 数据标准化
        2. 创建时序序列
        3. 分割训练测试集
        4. 构建并训练模型
        5. 评估模型性能
        
        参数：
        features_data (DataFrame): 准备好的特征数据
        
        返回：
        dict: 训练历史和评估结果
        """
        
        print("开始训练LSTM模型...")
        print(f"训练数据形状: {features_data.shape}")
        
        try:
            # 1. 数据标准化
            print("1. 标准化数据...")
            
            # fit_transform：计算均值和标准差，并进行标准化
            scaled_data = self.scaler.fit_transform(features_data)
            
            # 转换回DataFrame格式，保持列名和索引
            scaled_df = pd.DataFrame(
                scaled_data,
                columns=features_data.columns,
                index=features_data.index
            )
            
            print(f"   标准化后数据范围: [{scaled_data.min():.3f}, {scaled_data.max():.3f}]")
            
            # 2. 创建时序序列
            print("2. 创建时序序列...")
            X, y = self.create_sequences(scaled_df, target_column='Close')
            
            # 检查数据量是否足够
            min_samples = 100  # 最少需要100个样本
            if len(X) < min_samples:
                raise ValueError(f"训练数据不足：只有 {len(X)} 个样本，至少需要 {min_samples} 个")
            
            # 3. 分割训练测试集
            print("3. 分割数据集...")
            
            # 对于时间序列数据，不能随机分割，要保持时间顺序
            # 前80%用于训练，后20%用于测试
            split_index = int(len(X) * 0.8)
            
            X_train = X[:split_index]
            X_test = X[split_index:]
            y_train = y[:split_index]
            y_test = y[split_index:]
            
            print(f"   训练集: {X_train.shape[0]} 个样本")
            print(f"   测试集: {X_test.shape[0]} 个样本")
            
            # 4. 构建模型
            print("4. 构建神经网络...")
            
            # 输入形状：(时间步长, 特征数量)
            input_shape = (X.shape[1], X.shape[2])
            self.model = self.build_model(input_shape)
            
            # 5. 设置训练回调函数
            print("5. 设置训练回调...")
            
            # 早停：如果验证损失不再改善，提前停止训练
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',                           # 监控验证损失
                patience=LSTM_CONFIG['early_stopping_patience'], # 等待轮数
                restore_best_weights=True,                    # 恢复最佳权重
                verbose=1
            )
            
            # 学习率衰减：如果损失不再改善，减少学习率
            lr_reduction = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',                          # 监控验证损失
                patience=LSTM_CONFIG['lr_reduction_patience'], # 等待轮数
                factor=0.5,                                  # 衰减因子
                min_lr=1e-7,                                 # 最小学习率
                verbose=1
            )
            
            callbacks = [early_stopping, lr_reduction]
            
            # 6. 开始训练
            print("6. 开始训练模型...")
            print(f"   训练轮数: {self.epochs}")
            print(f"   批次大小: {self.batch_size}")
            
            # 训练模型
            history = self.model.fit(
                X_train, y_train,                    # 训练数据
                epochs=self.epochs,                  # 训练轮数
                batch_size=self.batch_size,          # 批次大小
                validation_data=(X_test, y_test),    # 验证数据
                callbacks=callbacks,                 # 回调函数
                verbose=1                            # 显示训练过程
            )
            
            # 7. 评估模型
            print("7. 评估模型性能...")
            
            # 在测试集上评估
            test_loss, test_mae = self.model.evaluate(X_test, y_test, verbose=0)
            
            # 预测测试集
            y_pred = self.model.predict(X_test, verbose=0)
            
            # 计算额外的评估指标
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            # 计算MAPE（平均绝对百分比误差）
            mape = np.mean(np.abs((y_test - y_pred.flatten()) / y_test)) * 100
            
            # 保存训练历史
            self.training_history = {
                'loss': history.history['loss'],
                'val_loss': history.history['val_loss'],
                'mae': history.history['mae'],
                'val_mae': history.history['val_mae'],
                'epochs_trained': len(history.history['loss'])
            }
            
            # 标记为已训练
            self.is_trained = True
            
            # 打印评估结果
            print("✅ 模型训练完成")
            print(f"   最终验证损失: {test_loss:.6f}")
            print(f"   平均绝对误差: {test_mae:.6f}")
            print(f"   均方误差: {mse:.6f}")
            print(f"   平均绝对百分比误差: {mape:.2f}%")
            print(f"   实际训练轮数: {len(history.history['loss'])}")
            
            return {
                'success': True,
                'test_loss': test_loss,
                'test_mae': test_mae,
                'mse': mse,
                'mape': mape,
                'epochs_trained': len(history.history['loss']),
                'history': self.training_history
            }
            
        except Exception as e:
            print(f"❌ 模型训练失败: {e}")
            self.is_trained = False
            return {
                'success': False,
                'error': str(e)
            }
    
    def predict_next_price(self, recent_features):
        """
        预测下一个交易日的价格
        
        参数：
        recent_features (DataFrame or np.array): 最近的特征数据
        
        返回：
        dict: 预测结果，包含价格和置信度
        """
        
        if not self.is_trained or self.model is None:
            raise ValueError("模型尚未训练，请先调用 train_model()")
        
        print("正在预测下一个交易日价格...")
        
        try:
            # 如果输入是DataFrame，转换为numpy数组
            if isinstance(recent_features, pd.DataFrame):
                # 验证特征列是否匹配
                if self.feature_columns and list(recent_features.columns) != self.feature_columns:
                    print("警告：特征列与训练时不匹配")
                
                features_array = recent_features.values
            else:
                features_array = recent_features
            
            # 标准化输入数据（使用训练时的参数）
            scaled_features = self.scaler.transform(features_array)
            
            # 检查数据长度是否足够
            if len(scaled_features) < self.sequence_length:
                raise ValueError(f"输入数据长度不足：需要至少 {self.sequence_length} 个数据点，"
                               f"但只有 {len(scaled_features)} 个")
            
            # 取最近的sequence_length个数据点
            input_sequence = scaled_features[-self.sequence_length:]
            
            # 重新整形为模型需要的格式：(1, sequence_length, features)
            input_sequence = input_sequence.reshape(1, self.sequence_length, -1)
            
            print(f"   输入序列形状: {input_sequence.shape}")
            
            # 使用模型进行预测
            prediction = self.model.predict(input_sequence, verbose=0)[0][0]
            
            # 计算置信度（基于最近的验证损失）
            if self.training_history:
                recent_val_loss = self.training_history['val_loss'][-5:]  # 最近5轮的验证损失
                avg_loss = np.mean(recent_val_loss)
                # 将损失转换为置信度（损失越小，置信度越高）
                confidence = max(0.1, min(0.95, 1.0 - avg_loss))
            else:
                confidence = 0.5  # 默认置信度
            
            print(f"✅ 预测完成")
            print(f"   预测价格: {prediction:.4f} (标准化后)")
            print(f"   置信度: {confidence:.3f}")
            
            return {
                'predicted_price': float(prediction),
                'confidence': float(confidence),
                'input_shape': input_sequence.shape,
                'model_used': 'LSTM'
            }
            
        except Exception as e:
            print(f"❌ 预测失败: {e}")
            raise ValueError(f"预测失败: {e}")
    
    def save_model(self, filepath):
        """
        保存训练好的模型
        
        参数：
        filepath (str): 保存路径
        """
        if not self.is_trained or self.model is None:
            raise ValueError("没有可保存的模型")
        
        try:
            # 保存模型
            self.model.save(f"{filepath}_model.h5")
            
            # 保存标准化器
            joblib.dump(self.scaler, f"{filepath}_scaler.pkl")
            
            # 保存其他信息
            model_info = {
                'feature_columns': self.feature_columns,
                'sequence_length': self.sequence_length,
                'training_history': self.training_history,
                'is_trained': self.is_trained
            }
            joblib.dump(model_info, f"{filepath}_info.pkl")
            
            print(f"✅ 模型保存成功: {filepath}")
            
        except Exception as e:
            print(f"❌ 模型保存失败: {e}")
    
    def load_model(self, filepath):
        """
        加载保存的模型
        
        参数：
        filepath (str): 模型文件路径
        """
        try:
            # 加载模型
            self.model = tf.keras.models.load_model(f"{filepath}_model.h5")
            
            # 加载标准化器
            self.scaler = joblib.load(f"{filepath}_scaler.pkl")
            
            # 加载其他信息
            model_info = joblib.load(f"{filepath}_info.pkl")
            self.feature_columns = model_info['feature_columns']
            self.sequence_length = model_info['sequence_length']
            self.training_history = model_info['training_history']
            self.is_trained = model_info['is_trained']
            
            print(f"✅ 模型加载成功: {filepath}")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise

# 模块测试代码
if __name__ == "__main__":
    print("=== 价格预测模块测试 ===")
    
    # 创建预测器实例
    predictor = StockPricePredictor()
    
    # 创建模拟数据进行测试
    print("\n--- 创建模拟测试数据 ---")
    
    # 生成模拟的股票数据
    import datetime
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)  # 设置随机种子，确保结果可重现
    
    # 模拟股价走势
    n_days = len(dates)
    prices = 100 + np.cumsum(np.random.randn(n_days) * 0.02)  # 随机游走
    volumes = np.random.randint(1000000, 10000000, n_days)
    
    # 创建模拟DataFrame
    mock_df = pd.DataFrame({
        'Open': prices * (1 + np.random.randn(n_days) * 0.01),
        'High': prices * (1 + abs(np.random.randn(n_days)) * 0.01),
        'Low': prices * (1 - abs(np.random.randn(n_days)) * 0.01),
        'Close': prices,
        'Volume': volumes
    }, index=dates)
    
    # 计算技术指标（简化版）
    mock_df['MA5'] = mock_df['Close'].rolling(window=5).mean()
    mock_df['MA20'] = mock_df['Close'].rolling(window=20).mean()
    mock_df['MA50'] = mock_df['Close'].rolling(window=50).mean()
    
    # RSI计算（简化）
    delta = mock_df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    mock_df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD计算
    exp1 = mock_df['Close'].ewm(span=12).mean()
    exp2 = mock_df['Close'].ewm(span=26).mean()
    mock_df['MACD'] = exp1 - exp2
    mock_df['MACD_Signal'] = mock_df['MACD'].ewm(span=9).mean()
    
    mock_df['Volatility'] = mock_df['Close'].rolling(window=20).std()
    mock_df['Price_Change'] = mock_df['Close'].pct_change()
    
    # 创建模拟的其他数据
    mock_stock_data = {'price_data': mock_df}
    mock_sentiment_data = {'overall_sentiment': 0.1, 'confidence': 0.7}
    mock_economic_data = {
        'GDP': [{'value': 2.1}, {'value': 2.3}],
        'Inflation': [{'value': 3.2}, {'value': 3.1}]
    }
    
    print(f"模拟数据创建完成，数据点数: {len(mock_df)}")
    
    # 测试特征准备
    print("\n--- 测试特征准备 ---")
    try:
        features = predictor.prepare_features(
            mock_stock_data, mock_sentiment_data, mock_economic_data
        )
        print(f"✅ 特征准备成功，特征维度: {features.shape}")
    except Exception as e:
        print(f"❌ 特征准备失败: {e}")
        exit(1)
    
    # 测试模型训练（使用少量轮数快速测试）
    print("\n--- 测试模型训练 ---")
    predictor.epochs = 5  # 减少训练轮数，快速测试
    
    try:
        result = predictor.train_model(features)
        if result['success']:
            print("✅ 模型训练测试成功")
        else:
            print(f"❌ 模型训练失败: {result['error']}")
    except Exception as e:
        print(f"❌ 模型训练测试失败: {e}")
    
    print("\n=== 测试完成 ===")