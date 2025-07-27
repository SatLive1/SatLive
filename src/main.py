"""MEO-LEO集群路由系统主程序 - 纯动态MEO/LEO版本"""
import sys
import os
# 将项目根目录添加到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 添加data目录到路径，以便导入data_loader
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'))
import random
import argparse
import json
from datetime import datetime

try:
    from config import Config
    from trainer import TrainingEnvironment
    from inferencer import ModelInferencer
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保在正确的目录下运行脚本")
    sys.exit(1)


def validate_required_dynamic_data_files(config: Config):
    """验证必需的动态数据文件是否存在 - 不提供数据生成功能，必须有数据"""
    print("\n=== 验证必需的动态数据文件 ===")

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root_dir, 'data')

    # 必需的动态MEO/LEO数据文件
    required_files = {
        'sat_positions_per_slot.json': '动态LEO位置数据',
        'meo_positions_per_slot.json': '动态MEO位置数据',
        'MEO_per_slot.json': 'LEO-MEO动态分配数据',
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
        print(f"\n❌ 错误：缺少必需的动态数据文件: {missing_files}")
        print("📋 必须提供以下文件于 data/ 目录：")
        for filename in missing_files:
            print(f"   • {filename}")
        print("\n💡 本系统仅支持动态MEO/LEO数据，不提供数据生成功能")
        raise FileNotFoundError(f"必需的动态数据文件缺失: {missing_files}")

    print("✅ 所有必需的动态数据文件验证通过")
    return True


def validate_dynamic_data_consistency(config: Config) -> bool:
    """验证动态MEO/LEO数据的完整性和一致性"""
    print("=== 验证动态数据一致性 ===")

    # 获取项目根目录
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root_dir, 'data')

    # 动态数据文件路径
    files = {
        'sat_positions': os.path.join(data_dir, 'sat_positions_per_slot.json'),
        'meo_positions': os.path.join(data_dir, 'meo_positions_per_slot.json'),
        'meo_assignments': os.path.join(data_dir, 'MEO_per_slot.json'),
        'config': os.path.join(data_dir, 'data.json')
    }

    try:
        # 加载所有动态数据文件
        with open(files['sat_positions'], 'r') as f:
            sat_positions_data = json.load(f)
        leo_slots = len(sat_positions_data)

        with open(files['meo_positions'], 'r') as f:
            meo_positions_data = json.load(f)
        meo_slots = len(meo_positions_data)

        with open(files['meo_assignments'], 'r') as f:
            meo_assignments_data = json.load(f)
        assignment_slots = len(meo_assignments_data)

        with open(files['config'], 'r') as f:
            config_data = json.load(f)

        # 获取配置参数
        num_leos = config_data.get('num_satellites', 1462)
        num_meos = config_data.get('num_meo_satellites', 32)

        print(f"   动态LEO时间槽数: {leo_slots}")
        print(f"   动态MEO时间槽数: {meo_slots}")
        print(f"   动态分配时间槽数: {assignment_slots}")
        print(f"   配置LEO数量: {num_leos}")
        print(f"   配置MEO数量: {num_meos}")

        # 验证时间槽数量一致性
        if leo_slots != meo_slots or leo_slots != assignment_slots:
            print(f"❌ 错误: 动态数据文件的时间槽数不一致")
            print(f"   LEO槽数: {leo_slots}, MEO槽数: {meo_slots}, 分配槽数: {assignment_slots}")
            return False

        # 验证至少有一个时间槽的数据
        if leo_slots == 0:
            print(f"❌ 错误: 没有找到任何时间槽的动态数据")
            return False

        # 验证MEO数量一致性
        if meo_positions_data and len(meo_positions_data[0]) != num_meos:
            actual_meo_count = len(meo_positions_data[0])
            print(f"❌ 错误: MEO位置数据中的数量({actual_meo_count})与配置不符({num_meos})")
            return False

        # 验证LEO数量一致性
        if sat_positions_data and len(sat_positions_data[0]) != num_leos:
            actual_leo_count = len(sat_positions_data[0])
            print(f"❌ 错误: LEO位置数据中的数量({actual_leo_count})与配置不符({num_leos})")
            return False

        # 验证动态数据元数据
        if 'dynamic_meo_metadata' in config_data:
            metadata = config_data['dynamic_meo_metadata']
            print(f"   动态MEO元数据: {metadata.get('constellation_type', 'unknown')} 星座")
        else:
            print("⚠️  警告: 缺少动态MEO元数据，但不影响运行")

        print("✅ 动态数据一致性验证通过")
        return True

    except Exception as e:
        print(f"❌ 动态数据验证失败: {e}")
        return False


def print_dynamic_system_info(config: Config):
    """打印动态MEO/LEO系统配置信息"""
    print("\n=== 动态MEO/LEO系统配置 ===")

    # 强制确保动态MEO启用（移除所有静态选项）
    config.update('network.enable_dynamic_meo', True)

    # 网络配置
    reassignment_enabled = config.get('network.enable_dynamic_meo_reassignment', False)
    reassignment_interval = config.get('network.meo_reassignment_interval', 5)

    print(f"系统类型: 🔄 纯动态MEO/LEO系统")
    print(f"动态重分配: {'启用' if reassignment_enabled else '禁用'}")
    if reassignment_enabled:
        print(f"重分配间隔: 每 {reassignment_interval} 个时间槽")

    # 路由配置
    inter_cluster_enabled = config.get('routing.inter_cluster_routing_enabled', True)
    k_paths = config.get('routing.k_paths', 3)

    print(f"跨集群路由: {'启用' if inter_cluster_enabled else '禁用'}")
    print(f"K路径数量: {k_paths}")

    # 动态特定奖励配置
    inter_cluster_reward = config.get('environment.reward_inter_cluster_success', 2.0)
    meo_adaptation_reward = config.get('environment.reward_meo_adaptation', 0.5)

    print(f"跨集群成功奖励: {inter_cluster_reward}")
    print(f"MEO适应奖励: {meo_adaptation_reward}")

    # 动态分析配置
    topology_analysis = config.get('training.enable_topology_analysis', True)
    dynamic_viz = config.get('output.enable_dynamic_visualization', True)

    print(f"动态拓扑分析: {'启用' if topology_analysis else '禁用'}")
    print(f"动态可视化: {'启用' if dynamic_viz else '禁用'}")


def main():
    """主函数 - 纯动态MEO/LEO版本"""
    parser = argparse.ArgumentParser(description='MEO-LEO集群路由系统 - 纯动态版本')
    parser.add_argument('--config', default='config.yaml', help='配置文件路径')
    parser.add_argument('--mode', choices=['train', 'inference', 'evaluate', 'data'], default='train',
                        help='运行模式: train=训练, inference=推理, evaluate=评估, data=数据加载演示')
    parser.add_argument('--model', help='模型文件路径（推理和评估模式需要）')
    parser.add_argument('--use-predict-data', action='store_true', default=True,
                        help='推理时使用预测数据集（默认）')
    parser.add_argument('--use-train-data', action='store_true',
                        help='推理时使用训练数据集')
    parser.add_argument('--output-dir', help='结果输出目录')
    parser.add_argument('--plot', action='store_true', default=True,
                        help='生成结果图表（默认开启）')

    args = parser.parse_args()

    print("🛰️ MEO-LEO集群路由系统 - 纯动态版本")
    print("=" * 50)

    # 加载配置
    try:
        config = Config(args.config)
        print(f"✅ 已加载配置文件: {args.config}")
    except FileNotFoundError:
        print(f"❌ 配置文件不存在: {args.config}")
        print("🚫 程序退出：请提供有效的配置文件")
        return
    except Exception as e:
        print(f"❌ 加载配置文件失败: {e}")
        return

    # 设置随机种子
    random_seed = config.get('simulation.random_seed', 42)
    random.seed(random_seed)
    print(f"🎲 随机种子: {random_seed}")

    # 强制启用动态MEO（完全移除静态选项）
    config.update('network.enable_dynamic_meo', True)
    print("🔄 系统强制运行在纯动态MEO/LEO模式")

    # 打印动态系统配置信息
    print_dynamic_system_info(config)

    # 验证必需的动态数据文件
    try:
        validate_required_dynamic_data_files(config)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("🚫 程序退出：必须提供完整的动态数据文件")
        return

    # 验证动态数据完整性
    if not validate_dynamic_data_consistency(config):
        print("❌ 动态数据验证失败，程序退出")
        return

    # 根据模式执行不同功能
    if args.mode == 'train':
        print("\n=== 开始训练 (纯动态MEO/LEO) ===")
        trainer = TrainingEnvironment(config)
        trainer.train()

    elif args.mode == 'inference':
        print("\n=== 开始推理 (纯动态MEO/LEO) ===")

        # 确定模型路径
        if args.model:
            model_path = args.model
        else:
            # 使用默认的最终模型路径
            results_path = config.get('output.results_path', 'results/')
            model_path = os.path.join(results_path, 'final_model.json')

        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            print("🚫 请先进行训练或指定正确的模型路径")
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
        print(f"🔍 使用{data_type}数据集进行动态推理")

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
        print(f"\n📊 动态系统模型综合质量评分: {quality_metrics.get('overall_quality', 0):.3f}")
        print(f"🔄 动态环境性能: {quality_metrics.get('dynamic_performance', 0):.3f}")

    elif args.mode == 'evaluate':
        print("\n=== 开始评估 (纯动态MEO/LEO) ===")

        # 确定模型路径
        if args.model:
            model_path = args.model
        else:
            results_path = config.get('output.results_path', 'results/')
            model_path = os.path.join(results_path, 'final_model.json')

        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            print("🚫 请先进行训练或指定正确的模型路径")
            return

        # 创建推理器
        inferencer = ModelInferencer(config)

        # 加载模型
        if not inferencer.load_trained_model(model_path):
            print("❌ 模型加载失败")
            return

        # 在训练数据和预测数据上都进行评估
        print("📈 在训练数据上评估动态性能...")
        train_results = inferencer.run_inference(use_predict_data=False)
        train_metrics = inferencer.performance_metrics.copy()

        print("\n📉 在预测数据上评估动态性能...")
        pred_results = inferencer.run_inference(use_predict_data=True)
        pred_metrics = inferencer.performance_metrics.copy()

        # 比较结果
        print("\n=== 动态系统评估结果比较 ===")
        print(f"训练集成功率: {train_metrics['success_rate']:.2%}")
        print(f"预测集成功率: {pred_metrics['success_rate']:.2%}")
        print(f"泛化差异: {abs(train_metrics['success_rate'] - pred_metrics['success_rate']):.2%}")

        print(f"训练集平均跳数: {train_metrics['average_hops']:.2f}")
        print(f"预测集平均跳数: {pred_metrics['average_hops']:.2f}")

        # 动态特定指标
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
            'system_type': 'pure_dynamic_meo_leo',  # 标记为纯动态系统
            'evaluation_timestamp': str(datetime.now())
        }

        eval_file = os.path.join(output_dir, 'evaluation_results_pure_dynamic.json')
        with open(eval_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2)

        print(f"\n💾 动态系统评估结果已保存到: {eval_file}")

        # 生成比较图表
        if args.plot:
            inferencer.plot_results(output_dir)

    elif args.mode == 'data':
        print("\n=== 动态数据加载演示 ===")
        try:
            from data_loader import load_complete_environment, print_environment_summary, validate_dynamic_meo_data
        except ImportError:
            import sys
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(root_dir, 'data')
            sys.path.insert(0, data_dir)
            try:
                from data_loader import load_complete_environment, print_environment_summary, validate_dynamic_meo_data
            except ImportError:
                print("❌ 无法导入data_loader模块")
                print("🚫 请确保data_loader.py存在于data目录中")
                return

        # 验证动态数据
        is_dynamic = validate_dynamic_meo_data()
        print(f"动态数据状态: {'✅ 有效' if is_dynamic else '❌ 无效或不存在'}")

        if not is_dynamic:
            print("❌ 系统仅支持动态MEO/LEO数据")
            print("🚫 程序退出：请提供有效的动态数据文件")
            return

        # 演示几个时间槽的动态数据加载
        demo_slots = [0, 2, 4]
        print(f"\n演示动态时间槽: {demo_slots}")

        for slot_id in demo_slots:
            try:
                leos, meos, data = load_complete_environment(slot_id, neighbors_dir="data/neighbors")
                print_environment_summary(leos, meos, slot_id)

                # 显示MEO移动信息
                if slot_id > 0:
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

    print("\n🎉 纯动态MEO/LEO系统执行完成!")


if __name__ == "__main__":
    main()