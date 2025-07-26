"""MEO-LEO集群路由系统主程序 - 支持动态MEO"""
import sys
import os
# 将项目根目录添加到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 添加data目录到路径，以便导入data_loader
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'))
import random
import argparse
import json

try:
    from config import Config
    from trainer import TrainingEnvironment
    from inferencer import ModelInferencer
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保在正确的目录下运行脚本")
    sys.exit(1)


def validate_dynamic_meo_setup(config: Config) -> bool:
    """验证动态MEO设置的完整性 - 修改为使用独立文件"""
    print("=== 验证动态MEO设置 ===")

    # 获取项目根目录
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root_dir, 'data')

    # 检查必要的独立数据文件
    required_files = {
        'sat_positions_per_slot.json': '卫星位置数据',
        'meo_positions_per_slot.json': 'MEO位置数据',
        'MEO_per_slot.json': 'LEO-MEO分配数据',
        'data.json': '主配置文件'
    }

    missing_files = []
    existing_files = {}

    for filename, description in required_files.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            print(f"✅ {description}: {filename}")
            existing_files[filename] = filepath
        else:
            print(f"❌ {description}缺失: {filename}")
            missing_files.append(filename)

    if missing_files:
        print(f"❌ 缺少必要文件: {missing_files}")
        return False

    try:
        # 检查独立文件的数据完整性

        # 1. 检查卫星位置数据
        with open(existing_files['sat_positions_per_slot.json'], 'r') as f:
            sat_positions_data = json.load(f)
        leo_slots = len(sat_positions_data)
        print(f"   LEO时间槽数: {leo_slots}")

        # 2. 检查MEO位置数据
        with open(existing_files['meo_positions_per_slot.json'], 'r') as f:
            meo_positions_data = json.load(f)
        meo_slots = len(meo_positions_data)
        print(f"   MEO时间槽数: {meo_slots}")

        # 3. 检查LEO-MEO分配数据
        with open(existing_files['MEO_per_slot.json'], 'r') as f:
            meo_assignments_data = json.load(f)
        assignment_slots = len(meo_assignments_data)
        print(f"   LEO-MEO分配时间槽数: {assignment_slots}")

        # 4. 检查主配置文件
        with open(existing_files['data.json'], 'r') as f:
            config_data = json.load(f)

        num_leos = config_data.get('num_satellites', 7)
        num_meos = config_data.get('num_meo_satellites', 3)
        print(f"   配置LEO数量: {num_leos}")
        print(f"   配置MEO数量: {num_meos}")

        # 验证数据一致性
        if leo_slots != meo_slots or leo_slots != assignment_slots:
            print(f"⚠️  警告: 各数据文件的时间槽数不匹配")
            print(f"   LEO槽数: {leo_slots}, MEO槽数: {meo_slots}, 分配槽数: {assignment_slots}")
            return False

        # 验证MEO数量一致性
        if meo_positions_data:
            actual_meo_count = len(meo_positions_data[0])
            if actual_meo_count != num_meos:
                print(f"⚠️  警告: MEO位置数据中的MEO数量({actual_meo_count})与配置不符({num_meos})")
                return False

        # 验证LEO数量一致性
        if sat_positions_data:
            actual_leo_count = len(sat_positions_data[0])
            if actual_leo_count != num_leos:
                print(f"⚠️  警告: LEO位置数据中的LEO数量({actual_leo_count})与配置不符({num_leos})")
                return False

        print("✅ 检测到动态MEO数据，启用动态MEO训练模式")
        return True

    except Exception as e:
        print(f"❌ 数据文件验证失败: {e}")
        return False


def print_dynamic_meo_info(config: Config):
    """打印动态MEO配置信息"""
    print("\n=== 动态MEO配置信息 ===")

    # 网络配置
    dynamic_meo_enabled = config.get('network.enable_dynamic_meo', True)
    reassignment_enabled = config.get('network.enable_dynamic_meo_reassignment', False)
    reassignment_interval = config.get('network.meo_reassignment_interval', 5)

    print(f"动态MEO支持: {'启用' if dynamic_meo_enabled else '禁用'}")
    print(f"动态重分配: {'启用' if reassignment_enabled else '禁用'}")
    if reassignment_enabled:
        print(f"重分配间隔: 每 {reassignment_interval} 个时间槽")

    # 路由配置
    inter_cluster_enabled = config.get('routing.inter_cluster_routing_enabled', True)
    k_paths = config.get('routing.k_paths', 3)
    edge_strategy = config.get('routing.edge_node_selection_strategy', 'advanced')

    print(f"跨集群路由: {'启用' if inter_cluster_enabled else '禁用'}")
    print(f"K路径数量: {k_paths}")
    print(f"边缘节点选择策略: {edge_strategy}")

    # 奖励配置
    inter_cluster_reward = config.get('environment.reward_inter_cluster_success', 2.0)
    meo_adaptation_reward = config.get('environment.reward_meo_adaptation', 0.5)

    print(f"跨集群成功奖励: {inter_cluster_reward}")
    print(f"MEO适应奖励: {meo_adaptation_reward}")

    # 分析配置
    topology_analysis = config.get('training.enable_topology_analysis', True)
    dynamic_viz = config.get('output.enable_dynamic_visualization', True)

    print(f"拓扑分析: {'启用' if topology_analysis else '禁用'}")
    print(f"动态可视化: {'启用' if dynamic_viz else '禁用'}")


def generate_sample_dynamic_meo_data(config: Config):
    """生成示例动态MEO数据 - 修改为创建独立文件"""
    print("\n=== 生成示例动态MEO数据 ===")

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root_dir, 'data')

    # 检查是否需要生成示例数据
    meo_pos_file = os.path.join(data_dir, 'meo_positions_per_slot.json')
    meo_assign_file = os.path.join(data_dir, 'MEO_per_slot.json')
    sat_pos_file = os.path.join(data_dir, 'sat_positions_per_slot.json')
    config_file = os.path.join(data_dir, 'data.json')

    if os.path.exists(meo_pos_file) and os.path.exists(meo_assign_file):
        print("✅ 已存在动态MEO数据，跳过生成")
        return True

    if not os.path.exists(config_file):
        print("❌ 主配置文件不存在，无法生成示例数据")
        return False

    try:
        # 读取主配置文件获取参数
        with open(config_file, 'r') as f:
            data = json.load(f)

        num_train_slots = data.get('num_train_slots', 10)
        num_meos = data.get('num_meo_satellites', 3)
        num_leos = data.get('num_satellites', 7)

        print(f"为 {num_train_slots} 个时间槽生成数据...")
        print(f"MEO数量: {num_meos}, LEO数量: {num_leos}")

        # 1. 生成MEO位置数据（如果不存在）
        if not os.path.exists(meo_pos_file):
            print("生成MEO位置数据...")
            meo_positions_per_slot = generate_meo_positions_data(num_train_slots, num_meos)

            with open(meo_pos_file, 'w') as f:
                json.dump(meo_positions_per_slot, f, indent=2)
            print(f"✅ 创建文件: {meo_pos_file}")

        # 2. 生成LEO-MEO分配数据（如果不存在）
        if not os.path.exists(meo_assign_file):
            print("生成LEO-MEO分配数据...")
            meo_assignments = generate_meo_assignments_data(num_train_slots, num_leos, num_meos)

            with open(meo_assign_file, 'w') as f:
                json.dump(meo_assignments, f, indent=2)
            print(f"✅ 创建文件: {meo_assign_file}")

        # 3. 检查LEO位置数据是否存在（如果不存在从data.json复制）
        if not os.path.exists(sat_pos_file):
            if 'sat_positions_per_slot' in data:
                print("从data.json复制LEO位置数据...")
                with open(sat_pos_file, 'w') as f:
                    json.dump(data['sat_positions_per_slot'], f, indent=2)
                print(f"✅ 创建文件: {sat_pos_file}")
            else:
                print("⚠️  警告: 未找到LEO位置数据")

        print("✅ 动态MEO数据已生成")
        return True

    except Exception as e:
        print(f"❌ 生成示例数据失败: {e}")
        return False


def generate_meo_positions_data(num_slots: int, num_meos: int):
    """生成MEO位置数据"""
    import random

    # MEO基础轨道位置
    base_positions = []
    for i in range(num_meos):
        base_lat = 45.0 + i * 10.0  # 分散在不同纬度
        base_lon = 45.0 + i * 10.0  # 分散在不同经度
        base_alt = 1000.0  # MEO高度
        base_positions.append([base_lat, base_lon, base_alt])

    meo_positions_per_slot = []

    for slot in range(num_slots):
        slot_positions = []
        for meo_id in range(num_meos):
            base_lat, base_lon, base_alt = base_positions[meo_id]

            # 模拟轨道运动
            orbit_offset = slot * 2.0
            movement_range = 3.0

            new_lat = base_lat + orbit_offset + random.uniform(-movement_range, movement_range)
            new_lon = base_lon + orbit_offset + random.uniform(-movement_range, movement_range)
            new_alt = base_alt + random.uniform(-50.0, 50.0)

            # 确保坐标在合理范围内
            new_lat = max(-90, min(90, new_lat))
            new_lon = new_lon % 360
            new_alt = max(800, min(1200, new_alt))

            slot_positions.append([new_lat, new_lon, new_alt])

        meo_positions_per_slot.append(slot_positions)

    return meo_positions_per_slot


def generate_meo_assignments_data(num_slots: int, num_leos: int, num_meos: int):
    """生成LEO-MEO分配数据"""
    meo_per_slot = []

    for slot in range(num_slots):
        # 简单的轮询分配
        leo_meo_assignments = [i % num_meos for i in range(num_leos)]

        slot_data = {
            "slot_id": slot,
            "leo_meo_assignments": leo_meo_assignments
        }
        meo_per_slot.append(slot_data)

    return meo_per_slot


def run_benchmark_comparison(config: Config, args):
    """运行动态MEO与静态MEO的基准对比"""
    print("\n=== 运行基准对比 ===")

    if not config.get('analysis.benchmark_against_static_meo', False):
        print("基准对比功能未启用")
        return

    print("此功能需要额外的实现...")
    # 这里可以实现动态MEO vs 静态MEO的性能对比


def main():
    """主函数 - 支持动态MEO"""
    parser = argparse.ArgumentParser(description='MEO-LEO集群路由系统 - 动态MEO版本')
    parser.add_argument('--config', default='config.yaml', help='配置文件路径')
    parser.add_argument('--mode', choices=['train', 'inference', 'evaluate', 'data', 'setup'], default='train',
                        help='运行模式: train=训练, inference=推理, evaluate=评估, data=数据加载演示, setup=设置向导')
    parser.add_argument('--model', help='模型文件路径（推理和评估模式需要）')
    parser.add_argument('--use-predict-data', action='store_true', default=True,
                        help='推理时使用预测数据集（默认）')
    parser.add_argument('--use-train-data', action='store_true',
                        help='推理时使用训练数据集')
    parser.add_argument('--output-dir', help='结果输出目录')
    parser.add_argument('--plot', action='store_true', default=True,
                        help='生成结果图表（默认开启）')
    parser.add_argument('--generate-sample-data', action='store_true',
                        help='生成示例动态MEO数据')
    parser.add_argument('--force-dynamic-meo', action='store_true',
                        help='强制启用动态MEO模式')
    parser.add_argument('--benchmark', action='store_true',
                        help='运行基准对比测试')

    args = parser.parse_args()

    print("🛰️ MEO-LEO集群路由系统 - 动态MEO版本")
    print("=" * 50)

    # 加载配置
    try:
        config = Config(args.config)
        print(f"✅ 已加载配置文件: {args.config}")
    except FileNotFoundError:
        print(f"❌ 配置文件不存在: {args.config}")
        return
    except Exception as e:
        print(f"❌ 加载配置文件失败: {e}")
        return

    # 设置随机种子
    random_seed = config.get('simulation.random_seed', 42)
    random.seed(random_seed)
    print(f"🎲 随机种子: {random_seed}")

    # 强制启用动态MEO（如果指定）
    if args.force_dynamic_meo:
        config.update('network.enable_dynamic_meo', True)
        print("🔄 强制启用动态MEO模式")

    # 打印动态MEO配置信息
    print_dynamic_meo_info(config)

    # 验证动态MEO设置
    if not validate_dynamic_meo_setup(config):
        print("⚠️  动态MEO设置验证失败")
        if not args.generate_sample_data:
            response = input("是否生成示例动态MEO数据？(y/N): ")
            if response.lower() == 'y':
                args.generate_sample_data = True

    # 生成示例数据（如果需要）
    if args.generate_sample_data:
        if not generate_sample_dynamic_meo_data(config):
            print("❌ 无法生成示例数据，程序退出")
            return

    # 根据模式执行不同功能
    if args.mode == 'setup':
        print("\n=== 设置向导 ===")
        print("动态MEO设置向导功能")
        # 这里可以实现交互式设置向导
        print("设置向导功能开发中...")

    elif args.mode == 'train':
        print("\n=== 开始训练 (动态MEO) ===")
        trainer = TrainingEnvironment(config)
        trainer.train()

    elif args.mode == 'inference':
        print("\n=== 开始推理 (动态MEO) ===")

        # 确定模型路径
        if args.model:
            model_path = args.model
        else:
            # 使用默认的最终模型路径
            results_path = config.get('output.results_path', 'results/')
            model_path = os.path.join(results_path, 'final_model.json')

        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            print("请先进行训练或指定正确的模型路径")
            return

        # 创建推理器
        inferencer = ModelInferencer(config)

        # 加载模型
        if not inferencer.load_trained_model(model_path):
            print("❌ 模型加载失败")
            return

        # 确定使用的数据集
        use_predict_data = not args.use_train_data  # 默认使用预测数据
        data_type = "预测" if use_predict_data else "训练"
        print(f"🔍 使用{data_type}数据集进行推理")

        # 运行推理
        results = inferencer.run_inference(use_predict_data=use_predict_data)

        # 保存结果
        output_dir = args.output_dir if args.output_dir else None
        inferencer.save_results(output_dir)

        # 生成图表
        if args.plot:
            inferencer.plot_results(output_dir)

        # 模型质量评估
        quality_metrics = inferencer.evaluate_model_quality()
        print(f"\n📊 模型综合质量评分: {quality_metrics.get('overall_quality', 0):.3f}")
        print(f"🔄 动态环境性能: {quality_metrics.get('dynamic_performance', 0):.3f}")

    elif args.mode == 'evaluate':
        print("\n=== 开始评估 (动态MEO) ===")

        # 确定模型路径
        if args.model:
            model_path = args.model
        else:
            results_path = config.get('output.results_path', 'results/')
            model_path = os.path.join(results_path, 'final_model.json')

        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            print("请先进行训练或指定正确的模型路径")
            return

        # 创建推理器
        inferencer = ModelInferencer(config)

        # 加载模型
        if not inferencer.load_trained_model(model_path):
            print("❌ 模型加载失败")
            return

        # 在训练数据和预测数据上都进行评估
        print("📈 在训练数据上评估...")
        train_results = inferencer.run_inference(use_predict_data=False)
        train_metrics = inferencer.performance_metrics.copy()

        print("\n📉 在预测数据上评估...")
        pred_results = inferencer.run_inference(use_predict_data=True)
        pred_metrics = inferencer.performance_metrics.copy()

        # 比较结果
        print("\n=== 评估结果比较 ===")
        print(f"训练集成功率: {train_metrics['success_rate']:.2%}")
        print(f"预测集成功率: {pred_metrics['success_rate']:.2%}")
        print(f"泛化差异: {abs(train_metrics['success_rate'] - pred_metrics['success_rate']):.2%}")

        print(f"训练集平均跳数: {train_metrics['average_hops']:.2f}")
        print(f"预测集平均跳数: {pred_metrics['average_hops']:.2f}")

        # 动态MEO特定指标
        if 'inter_cluster_success_rate' in train_metrics:
            print(f"训练集跨集群成功率: {train_metrics['inter_cluster_success_rate']:.2%}")
            print(f"预测集跨集群成功率: {pred_metrics['inter_cluster_success_rate']:.2%}")

        if 'average_meo_movement' in pred_metrics:
            print(f"平均MEO移动距离: {pred_metrics['average_meo_movement']:.2f}")

        # 保存评估结果
        output_dir = args.output_dir if args.output_dir else config.get('output.results_path', 'results/')
        os.makedirs(output_dir, exist_ok=True)

        evaluation_results = {
            'model_path': model_path,
            'train_metrics': train_metrics,
            'predict_metrics': pred_metrics,
            'generalization_gap': abs(train_metrics['success_rate'] - pred_metrics['success_rate']),
            'dynamic_meo_enabled': True,
            'evaluation_timestamp': str(datetime.now())
        }

        eval_file = os.path.join(output_dir, 'evaluation_results_dynamic_meo.json')
        with open(eval_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2)

        print(f"\n💾 评估结果已保存到: {eval_file}")

        # 生成比较图表
        if args.plot:
            inferencer.plot_results(output_dir)

    elif args.mode == 'data':
        print("\n=== 数据加载演示 (动态MEO) ===")
        # 导入data_loader模块
        try:
            from data_loader import load_complete_environment, print_environment_summary, validate_dynamic_meo_data
        except ImportError:
            # 如果导入失败，尝试绝对路径导入
            import sys
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(root_dir, 'data')
            sys.path.insert(0, data_dir)
            from ..data.data_loader import load_complete_environment, print_environment_summary, validate_dynamic_meo_data

        # 验证动态MEO数据
        is_dynamic = validate_dynamic_meo_data()
        print(f"动态MEO数据: {'✅ 有效' if is_dynamic else '❌ 无效或不存在'}")

        # 演示几个时间槽的数据加载
        demo_slots = [0, 2, 4]
        print(f"\n演示时间槽: {demo_slots}")

        for slot_id in demo_slots:
            try:
                leos, meos, data = load_complete_environment(slot_id, neighbors_dir="data/neighbors")
                print_environment_summary(leos, meos, slot_id)

                # 显示MEO移动信息（如果可用）
                if slot_id > 0 and is_dynamic:
                    prev_leos, prev_meos, _ = load_complete_environment(slot_id - 1, neighbors_dir="data/neighbors")
                    print(f"MEO移动信息 (从时间槽 {slot_id-1} 到 {slot_id}):")
                    for meo_id in meos:
                        if meo_id in prev_meos:
                            prev_pos = (prev_meos[meo_id].latitude, prev_meos[meo_id].longitude, prev_meos[meo_id].altitude)
                            curr_pos = (meos[meo_id].latitude, meos[meo_id].longitude, meos[meo_id].altitude)
                            distance = ((curr_pos[0] - prev_pos[0])**2 +
                                      (curr_pos[1] - prev_pos[1])**2 +
                                      (curr_pos[2] - prev_pos[2])**2)**0.5
                            print(f"  MEO {meo_id}: 移动距离 {distance:.2f}")

            except Exception as e:
                print(f"❌ 加载时间槽 {slot_id} 失败: {e}")

    else:
        print(f"❌ 未知的运行模式: {args.mode}")
        parser.print_help()
        return

    # 运行基准对比（如果启用）
    if args.benchmark:
        run_benchmark_comparison(config, args)

    print("\n🎉 程序执行完成!")


if __name__ == "__main__":
    from datetime import datetime
    main()