#!/usr/bin/env python3
"""
测试脚本 - 验证所有模块导入是否正常
"""
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试所有关键模块的导入"""
    print("开始测试模块导入...")
    
    try:
        print("1. 测试 config 模块...")
        from src.config import Config
        print("   ✅ config 导入成功")
    except Exception as e:
        print(f"   ❌ config 导入失败: {e}")
        return False
    
    try:
        print("2. 测试 satellites 模块...")
        from src.satellites import LEOSatellite, MEOSatellite
        print("   ✅ satellites 导入成功")
    except Exception as e:
        print(f"   ❌ satellites 导入失败: {e}")
        return False
    
    try:
        print("3. 测试 environment 模块...")
        from src.environment import distance, find_nearest_available_leo
        print("   ✅ environment 导入成功")
    except Exception as e:
        print(f"   ❌ environment 导入失败: {e}")
        return False
    
    try:
        print("4. 测试 rl_agent 模块...")
        from src.rl_agent import RLAgent
        print("   ✅ rl_agent 导入成功")
    except Exception as e:
        print(f"   ❌ rl_agent 导入失败: {e}")
        return False
    
    try:
        print("5. 测试 routing 模块...")
        from src.routing import calculate_geographic_distance
        print("   ✅ routing 导入成功")
    except Exception as e:
        print(f"   ❌ routing 导入失败: {e}")
        return False
    
    try:
        print("6. 测试 trainer 模块...")
        from src.trainer import TrainingEnvironment
        print("   ✅ trainer 导入成功")
    except Exception as e:
        print(f"   ❌ trainer 导入失败: {e}")
        return False
    
    try:
        print("7. 测试 inferencer 模块...")
        from src.inferencer import ModelInferencer
        print("   ✅ inferencer 导入成功")
    except Exception as e:
        print(f"   ❌ inferencer 导入失败: {e}")
        return False
    
    try:
        print("8. 测试 data_loader 模块...")
        from data.data_loader import load_complete_environment
        print("   ✅ data_loader 导入成功")
    except Exception as e:
        print(f"   ❌ data_loader 导入失败: {e}")
        return False
    
    print("\n🎉 所有模块导入测试通过！")
    return True

if __name__ == "__main__":
    success = test_imports()
    if not success:
        print("\n❌ 导入测试失败，请检查模块路径和依赖")
        sys.exit(1)
    else:
        print("\n✅ 可以开始运行主程序") 