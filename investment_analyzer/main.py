# main.py
"""
主程序入口文件

这是整个投资分析系统的启动文件
当你运行 streamlit run main.py 时，程序从这里开始执行

文件的作用：
1. 检查和安装必要的依赖库
2. 初始化系统环境
3. 启动Web界面应用
4. 处理启动过程中的错误

如何运行：
在命令行中执行：streamlit run main.py
"""

import sys              # 系统相关功能
import subprocess       # 子进程管理，用于安装库
import importlib        # 动态导入模块
import os               # 操作系统接口
from pathlib import Path  # 路径处理

def check_python_version():
    """
    检查Python版本
    
    确保Python版本满足系统要求
    系统需要Python 3.8或更高版本
    """
    print("正在检查Python版本...")
    
    # 获取当前Python版本
    version = sys.version_info
    min_version = (3, 8)  # 最低要求版本
    
    print(f"当前Python版本: {version.major}.{version.minor}.{version.micro}")
    
    # 检查版本是否满足要求
    if (version.major, version.minor) < min_version:
        print(f"❌ Python版本过低，需要Python {min_version[0]}.{min_version[1]} 或更高版本")
        print(f"   当前版本: {version.major}.{version.minor}")
        print("   请升级Python后重试")
        sys.exit(1)  # 退出程序
    
    print(f"✅ Python版本检查通过")

def install_package(package_name):
    """
    安装单个Python包
    
    参数：
    package_name (str): 要安装的包名
    
    返回：
    bool: 安装是否成功
    """
    try:
        print(f"  正在安装 {package_name}...")
        
        # 使用pip安装包
        # subprocess.check_call()会运行外部命令
        subprocess.check_call([
            sys.executable,    # 当前Python解释器路径
            "-m", "pip",       # 使用pip模块
            "install",         # 安装命令
            package_name,      # 包名
            "--quiet"          # 安静模式，减少输出
        ])
        
        print(f"  ✅ {package_name} 安装成功")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"  ❌ {package_name} 安装失败: {e}")
        return False
    except Exception as e:
        print(f"  ❌ {package_name} 安装异常: {e}")
        return False

def check_and_install_requirements():
    """
    检查并安装必需的依赖库
    
    这个函数会检查每个必需的库是否已安装
    如果没有安装，会自动尝试安装
    """
    print("正在检查依赖库...")
    
    # 定义必需的库列表
    # 格式：(导入名, 安装名, 描述)
    required_packages = [
        ("streamlit", "streamlit", "Web界面框架"),
        ("pandas", "pandas", "数据处理库"),
        ("numpy", "numpy", "数值计算库"),
        ("yfinance", "yfinance", "股票数据获取"),
        ("requests", "requests", "HTTP请求库"),
        ("plotly", "plotly", "交互式图表"),
        ("sklearn", "scikit-learn", "机器学习库"),
        ("tensorflow", "tensorflow", "深度学习框架"),
        ("transformers", "transformers", "自然语言处理"),
        ("joblib", "joblib", "模型保存加载")
    ]
    
    missing_packages = []    # 缺失的包列表
    failed_installs = []     # 安装失败的包列表
    
    # 检查每个包是否已安装
    for import_name, install_name, description in required_packages:
        try:
            # 尝试导入包
            importlib.import_module(import_name)
            print(f"  ✅ {description} ({import_name}) - 已安装")
            
        except ImportError:
            # 如果导入失败，说明包未安装
            print(f"  ❌ {description} ({import_name}) - 未安装")
            missing_packages.append((import_name, install_name, description))
    
    # 如果有缺失的包，尝试安装
    if missing_packages:
        print(f"\n发现 {len(missing_packages)} 个缺失的依赖库，正在安装...")
        
        for import_name, install_name, description in missing_packages:
            print(f"\n安装 {description}...")
            success = install_package(install_name)
            
            if not success:
                failed_installs.append((import_name, install_name, description))
        
        # 检查安装结果
        if failed_installs:
            print(f"\n❌ {len(failed_installs)} 个库安装失败:")
            for import_name, install_name, description in failed_installs:
                print(f"  - {description} ({install_name})")
            
            print("\n请手动安装失败的库:")
            for import_name, install_name, description in failed_installs:
                print(f"  pip install {install_name}")
            
            return False
        else:
            print(f"\n✅ 所有依赖库安装完成")
            return True
    
    else:
        print("✅ 所有依赖库已安装")
        return True

def setup_environment():
    """
    设置系统环境
    
    配置一些环境变量和设置，优化系统运行
    """
    print("正在设置系统环境...")
    
    # 1. 设置TensorFlow日志级别，减少警告信息
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # 2. 设置matplotlib后端（如果使用图表）
    os.environ['MPLBACKEND'] = 'Agg'
    
    # 3. 禁用一些警告
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # 4. 确保输出编码正确
    if sys.platform == 'win32':
        # Windows平台设置
        os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    print("✅ 系统环境设置完成")

def check_system_resources():
    """
    检查系统资源
    
    确保系统有足够的资源运行应用
    """
    print("正在检查系统资源...")
    
    try:
        import psutil  # 系统资源监控库
        
        # 检查内存
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)  # 转换为GB
        
        print(f"  可用内存: {available_gb:.1f} GB")
        
        if available_gb < 2:
            print("  ⚠️ 可用内存较少，可能影响AI模型性能")
        else:
            print("  ✅ 内存充足")
        
        # 检查磁盘空间
        disk = psutil.disk_usage('/')
        free_gb = disk.free / (1024**3)
        
        print(f"  可用磁盘空间: {free_gb:.1f} GB")
        
        if free_gb < 1:
            print("  ⚠️ 磁盘空间不足，可能影响模型保存")
        else:
            print("  ✅ 磁盘空间充足")
            
    except ImportError:
        print("  ⚠️ 无法检查系统资源（psutil未安装）")
    except Exception as e:
        print(f"  ⚠️ 系统资源检查失败: {e}")
    
    print("✅ 系统资源检查完成")

def display_startup_info():
    """
    显示启动信息
    
    向用户展示系统信息和使用说明
    """
    print("\n" + "="*60)
    print("🤖 智能投资决策助手")
    print("="*60)
    print("功能特点：")
    print("📈 技术分析 - 移动平均线、RSI、MACD等指标")
    print("🎭 情感分析 - 基于新闻的市场情感识别")
    print("🧠 AI预测 - LSTM神经网络价格预测")
    print("💡 智能策略 - 个性化投资建议生成")
    print("\n使用说明：")
    print("1. 在浏览器中打开显示的网址")
    print("2. 在左侧面板选择股票和风险偏好")
    print("3. 点击'开始智能分析'按钮")
    print("4. 等待分析完成，查看结果")
    print("\n⚠️  风险提示：投资有风险，建议仅供参考")
    print("="*60)

def import_and_run_app():
    """
    导入并运行主应用
    
    这里导入web_interface模块并启动应用
    """
    try:
        print("正在启动Web界面...")
        
        # 导入Web界面模块
        from web_interface import WebInterface
        
        # 创建应用实例
        app = WebInterface()
        
        # 运行应用
        # 注意：这个方法会启动Streamlit服务器
        app.run_app()
        
    except ImportError as e:
        print(f"❌ 导入Web界面模块失败: {e}")
        print("请检查web_interface.py文件是否存在且正确")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ 应用启动失败: {e}")
        print("请检查错误信息并重试")
        sys.exit(1)

def main():
    """
    主函数
    
    程序的主要入口点，协调所有启动步骤
    """
    try:
        # 1. 检查Python版本
        check_python_version()
        
        # 2. 检查并安装依赖
        if not check_and_install_requirements():
            print("❌ 依赖库安装失败，无法启动应用")
            print("请手动安装失败的库后重试")
            sys.exit(1)
        
        # 3. 设置环境
        setup_environment()
        
        # 4. 检查系统资源
        check_system_resources()
        
        # 5. 显示启动信息
        display_startup_info()
        
        # 6. 启动应用
        import_and_run_app()
        
    except KeyboardInterrupt:
        # 用户按Ctrl+C中断
        print("\n\n用户中断，程序退出")
        sys.exit(0)
        
    except Exception as e:
        # 其他未预期的错误
        print(f"\n❌ 程序启动失败: {e}")
        print("请检查错误信息并重试")
        sys.exit(1)

# 程序入口点
# 当直接运行这个文件时（python main.py 或 streamlit run main.py），
# __name__ 的值是 "__main__"，所以会执行main()函数
if __name__ == "__main__":
    main()