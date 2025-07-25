#!/usr/bin/env python3
"""
MEO-LEO集群路由系统完整示例
演示训练、推理和评估的完整流程
"""

import os
import sys
import subprocess
import json
import argparse
from pathlib import Path


def run_command(cmd, description):
    """运行命令并显示输出"""
    print(f"\n{'=' * 60}")
    print(f"执行: {description}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'=' * 60}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("警告信息:")
            print(result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False


def check_prerequisites():
    """检查运行前提条件"""
    print("检查前提条件...")

    # 检查必要的文件 - 根据实际目录结构
    required_files = [
        'config.yaml',
        'data/data.json'  # 数据文件在data文件夹中
    ]

    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)

    if missing_files:
        print(f"缺少必要文件: {', '.join(missing_files)}")
        print("提示：请检查以下文件是否存在：")
        print("  - config.yaml (在根目录)")
        print("  - data/data.json (在data文件夹中)")
        return False

    # 检查src目录
    if not os.path.exists('src'):
        print("缺少src目录，请确保在正确的项目根目录下运行")
        return False

    # 检查Python模块
    try:
        import yaml
        import numpy as np
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"缺少必要的Python包: {e}")
        print("请安装: pip install pyyaml numpy matplotlib")
        return False

    print("前提条件检查通过")
    return True


def create_demo_config():
    """创建演示用的配置文件"""
    config_content = """# MEO-LEO集群路由系统配置文件

# 数据配置
data:
  data_file: "data/data.json"  # 数据文件在data文件夹中

# 网络配置
network:
  num_time_slots: 50
  max_load_per_satellite: 10

# 仿真配置
simulation:
  random_seed: 42

# 强化学习代理配置
rl_agent:
  learning_rate: 0.1
  gamma: 0.9
  epsilon: 0.1
  epsilon_decay: 0.995
  epsilon_min: 0.01

# 训练配置
training:
  num_episodes: 100  # 演示用较小的值
  max_steps_per_episode: 50
  save_interval: 50

# 环境奖励配置
environment:
  reward_success: 10.0
  reward_failure: -5.0
  reward_timeout: -2.0
  reward_routing_success: 1.0
  reward_forwarding: 0.1
  reward_hop: -0.1
  reward_delay: -0.2
  reward_connection_lost: -1.0
  reward_load_balance: 1.0

# 输出配置
output:
  log_level: "INFO"
  log_file: "logs/demo.log"
  model_save_path: "models/"
  results_path: "results/"
  plot_results: true
"""

    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    with open('config.yaml', 'w', encoding='utf-8') as f:
        f.write(config_content)

    print("创建了配置文件: config.yaml")


def run_full_pipeline():
    """运行完整的训练和推理流程"""
    print("开始完整的MEO-LEO路由系统演示")

    # 1. 训练模型
    train_cmd = [
        sys.executable, '-m', 'src.main',
        '--config', 'config.yaml',
        '--mode', 'train'
    ]

    if not run_command(train_cmd, "训练RL模型"):
        print("训练失败，停止流程")
        return False

    # 2. 在预测数据上推理
    inference_cmd = [
        sys.executable, '-m', 'src.main',
        '--config', 'config.yaml',
        '--mode', 'inference',
        '--use-predict-data'
    ]

    if not run_command(inference_cmd, "在预测数据上推理"):
        print("推理失败，但继续后续步骤")

    # 3. 完整评估
    evaluate_cmd = [
        sys.executable, '-m', 'src.main',
        '--config', 'config.yaml',
        '--mode', 'evaluate'
    ]

    if not run_command(evaluate_cmd, "完整模型评估"):
        print("评估失败，但流程继续")

    return True


def show_results():
    """显示结果摘要"""
    print(f"\n{'=' * 60}")
    print("结果摘要")
    print(f"{'=' * 60}")

    results_dir = Path('results')

    # 显示训练统计
    train_stats_file = results_dir / 'training_stats.json'
    if train_stats_file.exists():
        with open(train_stats_file, 'r') as f:
            train_stats = json.load(f)

        print("\n训练结果:")
        if train_stats['episode_rewards']:
            final_reward = train_stats['episode_rewards'][-1]
            final_success_rate = train_stats['episode_success_rates'][-1]
            final_path_length = train_stats['episode_avg_path_lengths'][-1]

            print(f"  最终奖励: {final_reward:.2f}")
            print(f"  最终成功率: {final_success_rate:.2%}")
            print(f"  最终平均路径长度: {final_path_length:.2f}")

    # 显示推理结果
    inference_file = results_dir / 'performance_metrics.json'
    if inference_file.exists():
        with open(inference_file, 'r') as f:
            metrics = json.load(f)

        print("\n推理结果:")
        print(f"  总查询数: {metrics.get('total_queries', 0)}")
        print(f"  成功率: {metrics.get('success_rate', 0):.2%}")
        print(f"  平均跳数: {metrics.get('average_hops', 0):.2f}")
        print(f"  平均延迟: {metrics.get('average_delay', 0):.2f}")
        print(f"  平均推理时间: {metrics.get('average_inference_time_ms', 0):.2f} ms")

    # 显示评估结果
    eval_file = results_dir / 'evaluation_results.json'
    if eval_file.exists():
        with open(eval_file, 'r') as f:
            eval_results = json.load(f)

        print("\n模型评估:")
        train_metrics = eval_results.get('train_metrics', {})
        predict_metrics = eval_results.get('predict_metrics', {})

        print(f"  训练集成功率: {train_metrics.get('success_rate', 0):.2%}")
        print(f"  预测集成功率: {predict_metrics.get('success_rate', 0):.2%}")
        print(f"  泛化差异: {eval_results.get('generalization_gap', 0):.2%}")

    # 显示生成的文件
    print(f"\n生成的文件:")
    important_files = [
        'results/final_model.json',
        'results/training_stats.json',
        'results/performance_metrics.json',
        'results/training_results.png',
        'results/inference_analysis.png'
    ]

    for file_path in important_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"  ✓ {file_path} ({size} bytes)")
        else:
            print(f"  ✗ {file_path} (未生成)")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='MEO-LEO路由系统完整示例')
    parser.add_argument('--skip-check', action='store_true', help='跳过前提条件检查')
    parser.add_argument('--create-config', action='store_true', help='只创建配置文件')
    parser.add_argument('--show-results', action='store_true', help='只显示结果')

    args = parser.parse_args()

    if args.create_config:
        create_demo_config()
        return

    if args.show_results:
        show_results()
        return

    # 检查前提条件
    if not args.skip_check and not check_prerequisites():
        print("\n解决方案:")
        print("1. 如果没有config.yaml，运行: python run_example.py --create-config")
        print("2. 确保data/data.json文件存在")
        print("3. 确保在项目根目录下运行此脚本")
        return

    # 创建配置（如果不存在）
    if not os.path.exists('config.yaml'):
        create_demo_config()

    # 运行完整流程
    if run_full_pipeline():
        print(f"\n{'=' * 60}")
        print("完整流程执行成功！")
        print(f"{'=' * 60}")

        # 显示结果摘要
        show_results()

        print(f"\n检查 results/ 目录查看详细结果和图表")
    else:
        print("流程执行中遇到错误")


if __name__ == "__main__":
    main()