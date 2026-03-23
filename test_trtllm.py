#!/usr/bin/env python3
"""检查 TensorRT-LLM 安装状态"""

try:
    import tensorrt_llm
    print(f"✅ TensorRT-LLM 版本: {tensorrt_llm.__version__}")
    print(f"✅ 安装路径: {tensorrt_llm.__file__}")
    
    # 检查关键模块
    from tensorrt_llm import Builder
    print("✅ Builder 模块可导入")
    
    from tensorrt_llm.runtime import ModelRunner
    print("✅ ModelRunner 模块可导入")
    
    print("\n🎉 TensorRT-LLM 安装正常！")
    
except ImportError as e:
    print(f"❌ 导入失败: {e}")
except Exception as e:
    print(f"❌ 错误: {e}")
