#!/usr/bin/env python3
"""修复 build_engine.py 中的 TensorRT 旧版 API 兼容性问题."""

import sys

def fix_build_engine():
    filepath = 'inference/trt_llm/build_engine.py'
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"错误: 找不到文件 {filepath}")
        sys.exit(1)
    
    old_block = '''        build_cfg = builder.create_builder_config()
        build_cfg.max_workspace_size = 4 * 1024 * 1024 * 1024
        
        if dtype == "float16":
            build_cfg.set_flag(trt.BuilderFlag.FP16)
        
        build_cfg.max_batch_size = max_batch_size
        
        profile = builder.create_optimization_profile()
        profile.set_shape(
            "input_ids",
            min=(1, 1, num_quantizers),
            opt=(max(max_batch_size // 2, 1), max(model_max_seq_len // 2, 1), num_quantizers),
            max=(max_batch_size, model_max_seq_len, num_quantizers)
        )
        build_cfg.add_optimization_profile(profile)
        
        engine = builder.build_engine(network, build_cfg)
        
        if engine is None:
            logger.error("引擎构建失败")
            return False
        
        with open(output_path, 'wb') as f:
            f.write(engine.serialize())'''
    
    new_block = '''        build_cfg = builder.create_builder_config()
        try:
            build_cfg.max_workspace_size = 4 << 30
        except AttributeError:
            build_cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
        
        if dtype == "float16":
            build_cfg.set_flag(trt.BuilderFlag.FP16)
        
        profile = builder.create_optimization_profile()
        profile.set_shape(
            "input_ids",
            min=(1, 1, num_quantizers),
            opt=(max(max_batch_size // 2, 1), max(model_max_seq_len // 2, 1), num_quantizers),
            max=(max_batch_size, model_max_seq_len, num_quantizers)
        )
        build_cfg.add_optimization_profile(profile)
        
        try:
            engine = builder.build_engine(network, build_cfg)
        except AttributeError:
            serialized = builder.build_serialized_network(network, build_cfg)
            runtime = trt.Runtime(trt_logger)
            engine = runtime.deserialize_cuda_engine(serialized)
        
        if engine is None:
            logger.error("引擎构建失败")
            return False
        
        with open(output_path, 'wb') as f:
            try:
                f.write(engine.serialize())
            except TypeError:
                f.write(bytes(engine.serialize()))'''
    
    if old_block not in content:
        print("警告: 未找到需要替换的旧代码块，可能已经修复过或文件已被修改")
        print("尝试检查文件中是否仍存在 max_workspace_size...")
        if 'max_workspace_size' in content or 'max_batch_size' in content:
            print("文件中仍有旧 API，请手动检查")
            sys.exit(1)
        else:
            print("看起来已经修复过了，无需操作")
            sys.exit(0)
    
    content = content.replace(old_block, new_block)
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"✅ {filepath} 已修复")
    print("修复内容:")
    print("  - max_workspace_size -> 兼容 TensorRT 8.6+ (set_memory_pool_limit)")
    print("  - 移除 max_batch_size (EXPLICIT_BATCH 模式下已废弃)")
    print("  - build_engine -> 兼容 TensorRT 10.x (build_serialized_network)")
    print("  - engine.serialize() -> 兼容 TensorRT 10.x")

if __name__ == '__main__':
    fix_build_engine()
