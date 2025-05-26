# sentiment_analyzer.py
"""
情感分析模块

这个模块的作用：
1. 使用AI模型分析新闻文本的情感倾向
2. 将多条新闻的情感进行聚合
3. 为投资决策提供情感面的参考

什么是情感分析？
- 自然语言处理技术的一个分支
- 通过AI模型判断文本的情感色彩
- 在金融领域用于分析新闻、社交媒体对股价的影响
"""

# 导入transformers库，这是Hugging Face开发的AI模型库
# 包含了各种预训练的自然语言处理模型
from transformers import pipeline
import re                    # 正则表达式库，用于文本清理
import pandas as pd          # 数据处理
import numpy as np           # 数学计算
import warnings             # 警告管理

# 导入配置
from config import DATA_CONFIG

# 忽略transformers库的一些警告信息
warnings.filterwarnings('ignore')

class SentimentAnalyzer:
    """
    情感分析器类
    
    使用预训练的AI模型来分析文本情感
    特别针对金融新闻进行优化
    """
    
    def __init__(self):
        """
        初始化情感分析器
        
        加载预训练的情感分析模型
        优先使用金融专用模型，如果失败则使用通用模型
        """
        print("正在初始化情感分析器...")
        
        self.sentiment_pipeline = None  # 情感分析流水线
        self.model_name = None          # 使用的模型名称
        
        # 尝试加载金融专用的FinBERT模型
        try:
            print("  尝试加载FinBERT金融专用模型...")
            
            # FinBERT是专门为金融文本训练的BERT模型
            # 比通用模型更适合分析金融新闻的情感
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",              # 任务类型：情感分析
                model="ProsusAI/finbert",         # 模型名称
                tokenizer="ProsusAI/finbert"      # 分词器
            )
            self.model_name = "FinBERT"
            print("  ✅ FinBERT模型加载成功")
            
        except Exception as e:
            print(f"  ⚠️ FinBERT模型加载失败: {e}")
            print("  尝试使用默认情感分析模型...")
            
            try:
                # 备选方案：使用transformers的默认情感分析模型
                self.sentiment_pipeline = pipeline("sentiment-analysis")
                self.model_name = "Default"
                print("  ✅ 默认模型加载成功")
                
            except Exception as e:
                print(f"  ❌ 所有模型都加载失败: {e}")
                raise RuntimeError("无法加载任何情感分析模型")
        
        print(f"情感分析器初始化完成，使用模型: {self.model_name}")
    
    def analyze_news_sentiment(self, news_list):
        """
        分析新闻列表的情感
        
        参数：
        news_list (list): 新闻列表，每个元素是包含新闻信息的字典
        
        返回：
        dict: 聚合后的情感分析结果
        包含总体情感得分、置信度、新闻数量等信息
        """
        
        if not news_list:
            print("新闻列表为空，返回中性情感")
            return self._create_empty_result()
        
        print(f"正在分析 {len(news_list)} 条新闻的情感...")
        
        sentiment_scores = []  # 存储每条新闻的情感分析结果
        successful_analyses = 0  # 成功分析的新闻数量
        
        # 遍历每条新闻进行情感分析
        for i, news in enumerate(news_list):
            try:
                # 获取新闻标题
                text = news.get('title', '')
                
                # 检查文本是否有效
                if not text or len(text.strip()) == 0:
                    print(f"  新闻 {i+1}: 标题为空，跳过")
                    continue
                
                print(f"  分析新闻 {i+1}: {text[:50]}...")  # 显示前50个字符
                
                # 预处理文本
                cleaned_text = self._preprocess_text(text)
                
                if not cleaned_text:
                    print(f"    预处理后文本为空，跳过")
                    continue
                
                # 使用AI模型进行情感分析
                # sentiment_pipeline返回一个包含标签和置信度的字典
                result = self.sentiment_pipeline(cleaned_text)[0]
                
                # 整理分析结果
                sentiment_scores.append({
                    'text': text,                               # 原始文本
                    'cleaned_text': cleaned_text,               # 清理后的文本
                    'label': result['label'],                   # 情感标签(POSITIVE/NEGATIVE/NEUTRAL)
                    'score': result['score'],                   # 模型置信度(0-1)
                    'relevance': news.get('relevance', 1.0),   # 新闻相关性
                    'time': news.get('time', ''),              # 发布时间
                    'index': i                                  # 新闻索引
                })
                
                successful_analyses += 1
                print(f"    结果: {result['label']} (置信度: {result['score']:.3f})")
                
            except Exception as e:
                print(f"    分析新闻 {i+1} 时出错: {e}")
                continue
        
        print(f"✅ 情感分析完成，成功分析 {successful_analyses}/{len(news_list)} 条新闻")
        
        # 将多条新闻的情感结果聚合成一个总体得分
        return self._aggregate_sentiment(sentiment_scores)
    
    def _preprocess_text(self, text):
        """
        文本预处理
        
        为什么要预处理？
        - 清理掉特殊字符、多余空格等干扰信息
        - 让AI模型更好地理解文本内容
        - 避免因格式问题导致分析错误
        
        参数：
        text (str): 原始文本
        
        返回：
        str: 清理后的文本
        
        注意：方法名前的下划线表示这是私有方法
        """
        
        if not isinstance(text, str):
            return ""
        
        # 1. 移除HTML标签（如果有的话）
        # re.sub(pattern, replacement, string) 用replacement替换所有匹配pattern的部分
        text = re.sub(r'<[^>]+>', '', text)
        
        # 2. 移除URL链接
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # 3. 移除邮箱地址
        text = re.sub(r'\S+@\S+', '', text)
        
        # 4. 保留字母、数字、空格和基本标点符号
        # [^\w\s.,!?-] 表示匹配除了字母、数字、空格、逗号、句号、感叹号、问号、连字符以外的所有字符
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        # 5. 将多个连续空格替换为单个空格
        # \s+ 表示匹配一个或多个空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 6. 去除首尾空格
        text = text.strip()
        
        # 7. 限制文本长度（BERT模型通常有最大输入长度限制）
        max_length = 512  # BERT系列模型的典型最大长度
        if len(text) > max_length:
            text = text[:max_length]
            print(f"    文本过长，截断到 {max_length} 字符")
        
        return text
    
    def _aggregate_sentiment(self, sentiment_scores):
        """
        聚合多条新闻的情感分数
        
        为什么要聚合？
        - 单条新闻可能有偏差或噪音
        - 多条新闻的综合情感更能反映市场整体态度
        - 需要考虑每条新闻的置信度和相关性进行加权平均
        
        参数：
        sentiment_scores (list): 包含多条新闻情感分析结果的列表
        
        返回：
        dict: 聚合后的情感分析结果
        """
        
        # 如果没有成功分析的新闻，返回中性结果
        if not sentiment_scores:
            return self._create_empty_result()
        
        print("正在聚合情感分析结果...")
        
        # 初始化聚合计算的变量
        weighted_sentiment = 0  # 加权情感总和
        total_weight = 0        # 总权重
        
        # 统计各种情感标签的数量
        label_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        # 遍历每条新闻的分析结果
        for item in sentiment_scores:
            # 将文字标签转换为数值
            # 这样可以进行数学运算
            sentiment_value = self._convert_label_to_value(item['label'])
            
            # 计算这条新闻的权重
            # 权重 = 模型置信度 × 新闻相关性
            # 置信度高且相关性强的新闻对总体情感影响更大
            confidence = item['score']        # 模型置信度 (0-1)
            relevance = item['relevance']     # 新闻相关性 (0-1)
            weight = confidence * relevance
            
            # 累加加权情感值和权重
            weighted_sentiment += sentiment_value * weight
            total_weight += weight
            
            # 统计标签数量
            label = item['label'].lower()
            if label in label_counts:
                label_counts[label] += 1
        
        # 计算加权平均情感得分
        if total_weight > 0:
            overall_sentiment = weighted_sentiment / total_weight
        else:
            overall_sentiment = 0
        
        # 计算平均置信度
        avg_confidence = total_weight / len(sentiment_scores) if sentiment_scores else 0
        
        # 计算情感分布
        total_news = len(sentiment_scores)
        sentiment_distribution = {
            label: count / total_news for label, count in label_counts.items()
        }
        
        # 确定主导情感
        dominant_sentiment = max(label_counts, key=label_counts.get)
        
        print(f"  总体情感得分: {overall_sentiment:.3f}")
        print(f"  平均置信度: {avg_confidence:.3f}")
        print(f"  情感分布: {sentiment_distribution}")
        print(f"  主导情感: {dominant_sentiment}")
        
        # 返回聚合结果
        return {
            'overall_sentiment': overall_sentiment,       # 总体情感得分 (-1到1)
            'confidence': avg_confidence,                 # 平均置信度 (0到1)
            'news_count': total_news,                     # 分析的新闻数量
            'sentiment_distribution': sentiment_distribution,  # 情感分布
            'dominant_sentiment': dominant_sentiment,     # 主导情感
            'detailed_scores': sentiment_scores,          # 详细的每条新闻分析结果
            'model_used': self.model_name                # 使用的模型名称
        }
    
    def _convert_label_to_value(self, label):
        """
        将情感标签转换为数值
        
        参数：
        label (str): 情感标签 (POSITIVE/NEGATIVE/NEUTRAL等)
        
        返回：
        float: 对应的数值 (-1到1之间)
        """
        
        # 标准化标签（转为小写）
        label = label.lower()
        
        # 不同模型可能使用不同的标签格式
        if label in ['positive', 'pos']:
            return 1.0
        elif label in ['negative', 'neg']:
            return -1.0
        elif label in ['neutral']:
            return 0.0
        else:
            # 如果遇到未知标签，返回中性
            print(f"    未知情感标签: {label}，视为中性")
            return 0.0
    
    def _create_empty_result(self):
        """
        创建空的情感分析结果
        
        当没有新闻数据或分析失败时使用
        
        返回：
        dict: 空的情感分析结果
        """
        return {
            'overall_sentiment': 0.0,
            'confidence': 0.0,
            'news_count': 0,
            'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0},
            'dominant_sentiment': 'neutral',
            'detailed_scores': [],
            'model_used': self.model_name or 'None'
        }
    
    def analyze_single_text(self, text):
        """
        分析单条文本的情感
        
        这是一个公开方法，可以用于分析任何单条文本
        
        参数：
        text (str): 要分析的文本
        
        返回：
        dict: 情感分析结果
        """
        
        if not text:
            return None
        
        try:
            cleaned_text = self._preprocess_text(text)
            if not cleaned_text:
                return None
            
            result = self.sentiment_pipeline(cleaned_text)[0]
            
            return {
                'original_text': text,
                'cleaned_text': cleaned_text,
                'label': result['label'],
                'score': result['score'],
                'sentiment_value': self._convert_label_to_value(result['label'])
            }
            
        except Exception as e:
            print(f"分析文本时出错: {e}")
            return None

# 模块测试代码
if __name__ == "__main__":
    print("=== 情感分析模块测试 ===")
    
    # 创建情感分析器实例
    try:
        analyzer = SentimentAnalyzer()
        
        # 测试单条文本分析
        print("\n--- 测试单条文本分析 ---")
        test_texts = [
            "Apple stock soars to new record high on strong earnings!",
            "Market crash fears as inflation rises sharply",
            "The company reported quarterly results as expected"
        ]
        
        for text in test_texts:
            result = analyzer.analyze_single_text(text)
            if result:
                print(f"文本: {text}")
                print(f"情感: {result['label']} (置信度: {result['score']:.3f})")
                print(f"数值: {result['sentiment_value']}")
                print()
        
        # 测试新闻列表分析
        print("--- 测试新闻列表分析 ---")
        mock_news = [
            {
                'title': 'Apple reports record revenue growth',
                'relevance': 0.9,
                'time': '2024-01-01'
            },
            {
                'title': 'Tech stocks decline amid market uncertainty',
                'relevance': 0.7,
                'time': '2024-01-02'
            },
            {
                'title': 'Company announces new product launch',
                'relevance': 0.8,
                'time': '2024-01-03'
            }
        ]
        
        result = analyzer.analyze_news_sentiment(mock_news)
        print("聚合分析结果:")
        print(f"总体情感: {result['overall_sentiment']:.3f}")
        print(f"置信度: {result['confidence']:.3f}")
        print(f"新闻数量: {result['news_count']}")
        print(f"主导情感: {result['dominant_sentiment']}")
        
    except Exception as e:
        print(f"❌ 情感分析器初始化失败: {e}")
        print("这可能是因为网络问题或模型下载失败")
        print("请检查网络连接或尝试使用VPN")
    
    print("\n=== 测试完成 ===")