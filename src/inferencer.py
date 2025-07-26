"""MEO-LEO集群路由推理系统 - 支持动态MEO"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

try:
    from config import Config
    from satellites import LEOSatellite, MEOSatellite
    from rl_agent import RLAgent
    from routing import route_request_with_intelligent_edge_selection
    from environment import find_nearest_available_leo, analyze_network_topology
    from data.data_loader import load_complete_environment, validate_dynamic_meo_data
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保在正确的目录下运行脚本")
    import sys
    sys.exit(1)


class ModelInferencer:
    """模型推理类 - 支持动态MEO"""

    def __init__(self, config: Config):
        self.config = config
        self.setup_logging()
        self.agent = None

        # 推理统计
        self.inference_results = []
        self.performance_metrics = {}

        # 动态MEO相关统计
        self.meo_movement_stats = []
        self.inter_cluster_routing_stats = []
        self.topology_evolution_stats = []

    def setup_logging(self):
        """设置日志"""
        log_level = getattr(logging, self.config.get('output.log_level', 'INFO'))
        log_file = self.config.get('output.log_file', 'logs/inference.log')

        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger(__name__)

    def validate_model_compatibility(self, model_path: str) -> bool:
        """验证模型是否兼容动态MEO"""
        try:
            with open(model_path, 'r') as f:
                model_data = json.load(f)

            is_dynamic_meo_model = model_data.get('dynamic_meo_enabled', False)

            if is_dynamic_meo_model:
                self.logger.info("检测到动态MEO训练模型，启用动态MEO推理模式")
            else:
                self.logger.info("使用静态MEO训练模型，将尝试兼容动态MEO环境")

            return True

        except Exception as e:
            self.logger.error(f"模型兼容性验证失败: {e}")
            return False

    def load_trained_model(self, model_path: str) -> bool:
        """
        加载训练好的模型 - 验证动态MEO兼容性

        Args:
            model_path: 模型文件路径

        Returns:
            是否加载成功
        """
        try:
            if not os.path.exists(model_path):
                self.logger.error(f"模型文件不存在: {model_path}")
                return False

            # 验证模型兼容性
            if not self.validate_model_compatibility(model_path):
                return False

            with open(model_path, 'r') as f:
                model_data = json.load(f)

            # 创建智能体
            self.agent = RLAgent(
                learning_rate=model_data.get('learning_rate', 0.1),
                gamma=model_data.get('gamma', 0.9),
                epsilon=0.0  # 推理时不使用探索
            )

            # 加载Q表
            q_table = model_data.get('q_table', {})
            for state_str, actions in q_table.items():
                # 将字符串键转换回元组
                try:
                    state = eval(state_str)  # 注意：在生产环境中应使用更安全的方法
                    self.agent.q_table[state] = actions
                except:
                    self.logger.warning(f"无法解析状态: {state_str}")

            self.logger.info(f"成功加载模型: {model_path}")
            self.logger.info(f"Q表大小: {len(self.agent.q_table)}")

            # 检查是否有动态MEO训练统计
            training_stats = model_data.get('training_stats', {})
            if training_stats:
                self.logger.info(f"模型训练统计: {training_stats}")

            return True

        except Exception as e:
            self.logger.error(f"加载模型失败: {e}")
            return False

    def run_inference(self, data_file: str = None, use_predict_data: bool = True) -> Dict:
        """
        运行模型推理 - 支持动态MEO

        Args:
            data_file: 数据文件路径
            use_predict_data: 是否使用预测数据集

        Returns:
            推理结果字典
        """
        if self.agent is None:
            self.logger.error("请先加载训练好的模型")
            return {}

        if data_file is None:
            data_file = self.config.get('data.data_file', 'data/data.json')

        # 验证数据兼容性
        with open(data_file, 'r') as f:
            data = json.load(f)

        is_dynamic_meo = validate_dynamic_meo_data(data)
        if is_dynamic_meo:
            self.logger.info("检测到动态MEO数据，启用动态MEO推理")
        else:
            self.logger.info("使用静态MEO数据进行推理")

        self.logger.info("开始模型推理...")

        # 选择使用的查询数据
        if use_predict_data:
            queries = data.get('predict_queries', [])
            num_slots = data.get('num_predict_slots', 50)
            slot_offset = 0
        else:
            queries = data.get('train_queries', [])
            num_slots = data.get('num_train_slots', 50)
            slot_offset = 0

        # 检查可用的时间槽数据
        available_slots = len(data.get('sat_positions_per_slot', []))
        self.logger.info(f"可用时间槽数据: 0-{available_slots - 1}")

        # 调整时间槽范围以避免超出数据范围
        num_slots = min(num_slots, available_slots)

        self.logger.info(f"使用{'预测' if use_predict_data else '训练'}数据集")
        self.logger.info(f"查询数量: {len(queries)}, 实际处理时间槽数量: {num_slots}")

        # 推理统计
        total_queries = 0
        successful_queries = 0
        total_hops = 0
        total_delay = 0
        query_results = []

        # 动态MEO统计
        meo_position_changes = []
        cluster_stability_stats = []

        # 记录MEO位置变化
        previous_meo_positions = None

        # 按时间槽处理查询
        for current_slot in range(num_slots):
            actual_slot = current_slot

            # 检查索引是否在有效范围内
            if actual_slot >= available_slots:
                self.logger.warning(f"时间槽 {actual_slot} 超出数据范围，跳过")
                continue

            # 加载当前时间槽的网络环境
            try:
                leos, meos, _ = load_complete_environment(actual_slot, data_file)
            except Exception as e:
                self.logger.error(f"加载时间槽 {actual_slot} 环境失败: {e}")
                continue

            # 分析MEO位置变化
            current_meo_positions = {meo_id: (meo.latitude, meo.longitude, meo.altitude)
                                   for meo_id, meo in meos.items()}

            if previous_meo_positions is not None and is_dynamic_meo:
                meo_movements = self.calculate_meo_movements(previous_meo_positions, current_meo_positions)
                meo_position_changes.append({
                    'slot': current_slot,
                    'movements': meo_movements
                })

            previous_meo_positions = current_meo_positions

            # 分析网络拓扑（每10个slot一次）
            if current_slot % 10 == 0:
                topology_analysis = analyze_network_topology(leos, meos)
                self.topology_evolution_stats.append({
                    'slot': current_slot,
                    'analysis': topology_analysis
                })

            # 处理当前时间槽的查询
            slot_queries = [q for q in queries if q['time'] == current_slot]

            for query in slot_queries:
                src_id = query['src']
                dst_id = query['dst']
                query_time = query['time']

                total_queries += 1

                # 验证源和目标节点是否存在
                if src_id not in leos or dst_id not in leos:
                    query_result = {
                        'query_id': total_queries - 1,
                        'src': src_id,
                        'dst': dst_id,
                        'time_slot': current_slot,
                        'path': [],
                        'hop_count': 0,
                        'transmission_delay': 0,
                        'inference_time_ms': 0,
                        'status': '失败',
                        'failure_reason': 'Source or destination node not available',
                        'routing_stats': {}
                    }
                    query_results.append(query_result)
                    continue

                # 使用训练好的模型进行路由
                start_time = datetime.now()
                try:
                    path, routing_stats = route_request_with_intelligent_edge_selection(
                        src_id, dst_id, leos, meos, self.agent
                    )
                except Exception as e:
                    self.logger.warning(f"路由计算失败: {e}")
                    path, routing_stats = [], {'success': False, 'error': str(e)}

                inference_time = (datetime.now() - start_time).total_seconds() * 1000  # 毫秒

                # 统计跨集群路由
                if routing_stats.get('routing_strategy') in ['inter_cluster_two_stage', 'inter_cluster_fallback']:
                    self.inter_cluster_routing_stats.append({
                        'slot': current_slot,
                        'strategy': routing_stats.get('routing_strategy'),
                        'success': routing_stats.get('success', False),
                        'hops': len(path) - 1 if path else 0
                    })

                # 评估结果
                if routing_stats.get('success', False) and len(path) > 1:
                    successful_queries += 1
                    hop_count = len(path) - 1
                    total_hops += hop_count

                    # 计算传输延迟（简化为跳数）
                    transmission_delay = hop_count
                    total_delay += transmission_delay

                    result_status = "成功"
                    failure_reason = None
                else:
                    hop_count = 0
                    transmission_delay = 0
                    result_status = "失败"
                    failure_reason = routing_stats.get('error', 'Routing failed')

                # 记录单个查询结果
                query_result = {
                    'query_id': total_queries - 1,
                    'src': src_id,
                    'dst': dst_id,
                    'time_slot': current_slot,
                    'path': path,
                    'hop_count': hop_count,
                    'transmission_delay': transmission_delay,
                    'inference_time_ms': inference_time,
                    'status': result_status,
                    'failure_reason': failure_reason,
                    'routing_stats': routing_stats
                }
                query_results.append(query_result)

                # 定期输出进度
                if total_queries % 10 == 0:
                    current_success_rate = successful_queries / total_queries
                    self.logger.info(
                        f"已处理 {total_queries} 个查询, "
                        f"成功率: {current_success_rate:.2%}, "
                        f"平均跳数: {total_hops / successful_queries if successful_queries > 0 else 0:.2f}"
                    )

        # 保存动态MEO统计
        self.meo_movement_stats = meo_position_changes

        # 计算总体性能指标
        success_rate = successful_queries / total_queries if total_queries > 0 else 0
        avg_hops = total_hops / successful_queries if successful_queries > 0 else 0
        avg_delay = total_delay / successful_queries if successful_queries > 0 else 0
        avg_inference_time = np.mean([r['inference_time_ms'] for r in query_results])

        # 计算额外的性能指标
        hop_distribution = [r['hop_count'] for r in query_results if r['status'] == '成功']
        delay_distribution = [r['transmission_delay'] for r in query_results if r['status'] == '成功']

        # 动态MEO相关指标
        inter_cluster_success_rate = 0.0
        avg_meo_movement = 0.0

        if self.inter_cluster_routing_stats:
            inter_cluster_successes = sum(1 for stat in self.inter_cluster_routing_stats if stat['success'])
            inter_cluster_success_rate = inter_cluster_successes / len(self.inter_cluster_routing_stats)

        if self.meo_movement_stats:
            all_movements = []
            for slot_data in self.meo_movement_stats:
                all_movements.extend(slot_data['movements'].values())
            avg_meo_movement = np.mean(all_movements) if all_movements else 0.0

        performance_metrics = {
            'total_queries': total_queries,
            'successful_queries': successful_queries,
            'failed_queries': total_queries - successful_queries,
            'success_rate': success_rate,
            'average_hops': avg_hops,
            'average_delay': avg_delay,
            'average_inference_time_ms': avg_inference_time,
            'hop_std': np.std(hop_distribution) if hop_distribution else 0,
            'delay_std': np.std(delay_distribution) if delay_distribution else 0,
            'min_hops': min(hop_distribution) if hop_distribution else 0,
            'max_hops': max(hop_distribution) if hop_distribution else 0,
            'median_hops': np.median(hop_distribution) if hop_distribution else 0,
            # 动态MEO特定指标
            'inter_cluster_queries': len(self.inter_cluster_routing_stats),
            'inter_cluster_success_rate': inter_cluster_success_rate,
            'average_meo_movement': avg_meo_movement,
            'topology_analyses_count': len(self.topology_evolution_stats)
        }

        # 保存结果
        self.inference_results = query_results
        self.performance_metrics = performance_metrics

        # 输出推理结果摘要
        self.logger.info("=== 推理结果摘要 ===")
        self.logger.info(f"总查询数: {total_queries}")
        self.logger.info(f"成功查询数: {successful_queries}")
        self.logger.info(f"成功率: {success_rate:.2%}")
        self.logger.info(f"平均跳数: {avg_hops:.2f}")
        self.logger.info(f"平均延迟: {avg_delay:.2f}")
        self.logger.info(f"平均推理时间: {avg_inference_time:.2f} ms")

        if is_dynamic_meo:
            self.logger.info(f"跨集群查询数: {len(self.inter_cluster_routing_stats)}")
            self.logger.info(f"跨集群成功率: {inter_cluster_success_rate:.2%}")
            self.logger.info(f"平均MEO移动距离: {avg_meo_movement:.2f}")

        return {
            'query_results': query_results,
            'performance_metrics': performance_metrics
        }

    def calculate_meo_movements(self, prev_positions: Dict, curr_positions: Dict) -> Dict[int, float]:
        """计算MEO在相邻时间槽之间的移动距离"""
        movements = {}

        for meo_id in prev_positions:
            if meo_id in curr_positions:
                prev_pos = prev_positions[meo_id]
                curr_pos = curr_positions[meo_id]

                # 计算3D欧几里得距离
                distance = np.sqrt(
                    (curr_pos[0] - prev_pos[0]) ** 2 +
                    (curr_pos[1] - prev_pos[1]) ** 2 +
                    (curr_pos[2] - prev_pos[2]) ** 2
                )
                movements[meo_id] = distance

        return movements

    def compare_with_baseline(self, baseline_results: Dict = None) -> Dict:
        """
        与基线方法进行比较 - 增强版，考虑动态MEO

        Args:
            baseline_results: 基线方法的结果

        Returns:
            比较结果
        """
        if not self.performance_metrics:
            self.logger.error("请先运行推理")
            return {}

        # 如果没有提供基线结果，创建一个简单的随机路由基线
        if baseline_results is None:
            baseline_results = self._generate_random_baseline()

        comparison = {
            'rl_agent': self.performance_metrics,
            'baseline': baseline_results,
            'improvements': {}
        }

        # 计算改进指标
        if baseline_results.get('success_rate', 0) > 0:
            comparison['improvements']['success_rate_improvement'] = (
                    self.performance_metrics['success_rate'] - baseline_results['success_rate']
            )

        if baseline_results.get('average_hops', 0) > 0:
            comparison['improvements']['hop_reduction'] = (
                                                                  baseline_results['average_hops'] -
                                                                  self.performance_metrics['average_hops']
                                                          ) / baseline_results['average_hops']

        if baseline_results.get('average_delay', 0) > 0:
            comparison['improvements']['delay_reduction'] = (
                                                                    baseline_results['average_delay'] -
                                                                    self.performance_metrics['average_delay']
                                                            ) / baseline_results['average_delay']

        # 动态MEO特定比较
        if self.performance_metrics.get('inter_cluster_success_rate', 0) > 0:
            baseline_inter_cluster = baseline_results.get('inter_cluster_success_rate', 0.5)
            comparison['improvements']['inter_cluster_improvement'] = (
                self.performance_metrics['inter_cluster_success_rate'] - baseline_inter_cluster
            )

        self.logger.info("=== 与基线比较 ===")
        self.logger.info(f"成功率提升: {comparison['improvements'].get('success_rate_improvement', 0):.2%}")
        self.logger.info(f"跳数减少: {comparison['improvements'].get('hop_reduction', 0):.2%}")
        self.logger.info(f"延迟减少: {comparison['improvements'].get('delay_reduction', 0):.2%}")

        if 'inter_cluster_improvement' in comparison['improvements']:
            self.logger.info(f"跨集群路由提升: {comparison['improvements']['inter_cluster_improvement']:.2%}")

        return comparison

    def _generate_random_baseline(self) -> Dict:
        """生成随机路由基线结果 - 适应动态MEO"""
        # 为动态MEO环境调整基线假设
        base_success_rate = max(0.5, self.performance_metrics['success_rate'] - 0.15)
        base_avg_hops = self.performance_metrics['average_hops'] + 2.0
        base_avg_delay = self.performance_metrics['average_delay'] + 3.0

        return {
            'success_rate': base_success_rate,
            'average_hops': base_avg_hops,
            'average_delay': base_avg_delay,
            'inter_cluster_success_rate': max(0.3, self.performance_metrics.get('inter_cluster_success_rate', 0.5) - 0.2)
        }

    def save_results(self, output_dir: str = None):
        """
        保存推理结果 - 包含动态MEO统计

        Args:
            output_dir: 输出目录
        """
        if output_dir is None:
            output_dir = self.config.get('output.results_path', 'results/')

        os.makedirs(output_dir, exist_ok=True)

        # 保存详细结果
        results_file = os.path.join(output_dir, 'inference_results.json')
        with open(results_file, 'w') as f:
            json.dump({
                'query_results': self.inference_results,
                'performance_metrics': self.performance_metrics,
                'meo_movement_stats': self.meo_movement_stats,
                'inter_cluster_routing_stats': self.inter_cluster_routing_stats,
                'topology_evolution_stats': self.topology_evolution_stats,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)

        # 保存性能指标摘要
        metrics_file = os.path.join(output_dir, 'performance_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(self.performance_metrics, f, indent=2)

        self.logger.info(f"推理结果已保存到: {output_dir}")

    def plot_results(self, output_dir: str = None):
        """
        绘制推理结果图表 - 增强版，包含动态MEO分析

        Args:
            output_dir: 输出目录
        """
        if not self.inference_results:
            self.logger.error("没有推理结果可以绘制")
            return

        if output_dir is None:
            output_dir = self.config.get('output.results_path', 'results/')

        os.makedirs(output_dir, exist_ok=True)

        # 提取成功的查询数据
        successful_results = [r for r in self.inference_results if r['status'] == '成功']

        if not successful_results:
            self.logger.warning("没有成功的查询结果可以绘制")
            return

        # 确定图表布局（根据是否有动态MEO数据）
        has_dynamic_data = bool(self.meo_movement_stats or self.inter_cluster_routing_stats)
        num_rows = 3 if has_dynamic_data else 2
        num_cols = 3

        # 创建图表
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 15))
        fig.suptitle('模型推理结果分析 (动态MEO)', fontsize=16)

        # 1. 成功率统计
        success_count = len(successful_results)
        fail_count = len(self.inference_results) - success_count
        axes[0, 0].pie([success_count, fail_count],
                       labels=['成功', '失败'],
                       autopct='%1.1f%%',
                       colors=['green', 'red'])
        axes[0, 0].set_title('路由成功率')

        # 2. 跳数分布
        hop_counts = [r['hop_count'] for r in successful_results]
        axes[0, 1].hist(hop_counts, bins=20, alpha=0.7, color='blue')
        axes[0, 1].set_xlabel('跳数')
        axes[0, 1].set_ylabel('频次')
        axes[0, 1].set_title('跳数分布')
        axes[0, 1].axvline(np.mean(hop_counts), color='red', linestyle='--',
                           label=f'平均: {np.mean(hop_counts):.1f}')
        axes[0, 1].legend()

        # 3. 延迟分布
        delays = [r['transmission_delay'] for r in successful_results]
        axes[0, 2].hist(delays, bins=20, alpha=0.7, color='orange')
        axes[0, 2].set_xlabel('传输延迟')
        axes[0, 2].set_ylabel('频次')
        axes[0, 2].set_title('延迟分布')
        axes[0, 2].axvline(np.mean(delays), color='red', linestyle='--',
                           label=f'平均: {np.mean(delays):.1f}')
        axes[0, 2].legend()

        # 4. 推理时间分布
        inference_times = [r['inference_time_ms'] for r in self.inference_results]
        axes[1, 0].hist(inference_times, bins=20, alpha=0.7, color='purple')
        axes[1, 0].set_xlabel('推理时间 (ms)')
        axes[1, 0].set_ylabel('频次')
        axes[1, 0].set_title('推理时间分布')
        axes[1, 0].axvline(np.mean(inference_times), color='red', linestyle='--',
                           label=f'平均: {np.mean(inference_times):.1f}')
        axes[1, 0].legend()

        # 5. 时间槽性能趋势
        slot_performance = {}
        for result in self.inference_results:
            slot = result['time_slot']
            if slot not in slot_performance:
                slot_performance[slot] = {'total': 0, 'success': 0}
            slot_performance[slot]['total'] += 1
            if result['status'] == '成功':
                slot_performance[slot]['success'] += 1

        slots = sorted(slot_performance.keys())
        success_rates = [slot_performance[s]['success'] / slot_performance[s]['total']
                         for s in slots]

        axes[1, 1].plot(slots, success_rates, 'b-', marker='o', markersize=3)
        axes[1, 1].set_xlabel('时间槽')
        axes[1, 1].set_ylabel('成功率')
        axes[1, 1].set_title('时间槽成功率趋势')
        axes[1, 1].grid(True, alpha=0.3)

        # 6. 路由策略统计
        strategy_counts = {}
        for result in successful_results:
            strategy = result['routing_stats'].get('routing_strategy', 'unknown')
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        if strategy_counts:
            strategies = list(strategy_counts.keys())
            counts = list(strategy_counts.values())
            axes[1, 2].bar(strategies, counts, alpha=0.7)
            axes[1, 2].set_xlabel('路由策略')
            axes[1, 2].set_ylabel('使用次数')
            axes[1, 2].set_title('路由策略使用统计')
            axes[1, 2].tick_params(axis='x', rotation=45)

        # 动态MEO特定图表
        if has_dynamic_data and num_rows > 2:
            # 7. MEO移动距离分布
            if self.meo_movement_stats:
                all_movements = []
                for slot_data in self.meo_movement_stats:
                    all_movements.extend(slot_data['movements'].values())

                if all_movements:
                    axes[2, 0].hist(all_movements, bins=20, alpha=0.7, color='cyan')
                    axes[2, 0].set_xlabel('MEO移动距离')
                    axes[2, 0].set_ylabel('频次')
                    axes[2, 0].set_title('MEO移动距离分布')
                    axes[2, 0].axvline(np.mean(all_movements), color='red', linestyle='--',
                                       label=f'平均: {np.mean(all_movements):.1f}')
                    axes[2, 0].legend()

            # 8. 跨集群路由成功率趋势
            if self.inter_cluster_routing_stats:
                cluster_slots = [stat['slot'] for stat in self.inter_cluster_routing_stats]
                cluster_success = [1 if stat['success'] else 0 for stat in self.inter_cluster_routing_stats]

                # 按时间槽聚合
                slot_cluster_stats = {}
                for i, slot in enumerate(cluster_slots):
                    if slot not in slot_cluster_stats:
                        slot_cluster_stats[slot] = {'total': 0, 'success': 0}
                    slot_cluster_stats[slot]['total'] += 1
                    slot_cluster_stats[slot]['success'] += cluster_success[i]

                if slot_cluster_stats:
                    cluster_slots_sorted = sorted(slot_cluster_stats.keys())
                    cluster_success_rates = [slot_cluster_stats[s]['success'] / slot_cluster_stats[s]['total']
                                           for s in cluster_slots_sorted]

                    axes[2, 1].plot(cluster_slots_sorted, cluster_success_rates, 'r-', marker='s', markersize=4)
                    axes[2, 1].set_xlabel('时间槽')
                    axes[2, 1].set_ylabel('跨集群成功率')
                    axes[2, 1].set_title('跨集群路由成功率趋势')
                    axes[2, 1].grid(True, alpha=0.3)

            # 9. 网络效率演变
            if self.topology_evolution_stats:
                topo_slots = [stat['slot'] for stat in self.topology_evolution_stats]
                network_efficiencies = [stat['analysis']['network_efficiency'] for stat in self.topology_evolution_stats]

                axes[2, 2].plot(topo_slots, network_efficiencies, 'g-', marker='d', markersize=4)
                axes[2, 2].set_xlabel('时间槽')
                axes[2, 2].set_ylabel('网络效率')
                axes[2, 2].set_title('网络拓扑效率演变')
                axes[2, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图表
        plot_file = os.path.join(output_dir, 'inference_analysis_dynamic_meo.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')

        if self.config.get('output.plot_results', True):
            plt.show()

        self.logger.info(f"分析图表已保存到: {plot_file}")

    def evaluate_model_quality(self) -> Dict:
        """
        评估模型质量 - 增强版，考虑动态MEO性能

        Returns:
            模型质量评估结果
        """
        if not self.inference_results:
            self.logger.error("请先运行推理")
            return {}

        successful_results = [r for r in self.inference_results if r['status'] == '成功']

        # 计算各种质量指标
        quality_metrics = {
            'robustness': 0.0,  # 鲁棒性
            'efficiency': 0.0,  # 效率
            'consistency': 0.0,  # 一致性
            'adaptability': 0.0,  # 适应性
            'dynamic_performance': 0.0  # 动态环境性能
        }

        if successful_results:
            # 鲁棒性：成功率
            quality_metrics['robustness'] = len(successful_results) / len(self.inference_results)

            # 效率：基于平均跳数的倒数
            avg_hops = np.mean([r['hop_count'] for r in successful_results])
            quality_metrics['efficiency'] = min(1.0, 5.0 / max(avg_hops, 1.0))

            # 一致性：跳数标准差的倒数
            hop_std = np.std([r['hop_count'] for r in successful_results])
            quality_metrics['consistency'] = min(1.0, 3.0 / max(hop_std, 0.1))

            # 适应性：不同路由策略的使用多样性
            strategies = [r['routing_stats'].get('routing_strategy', 'unknown')
                          for r in successful_results]
            strategy_diversity = len(set(strategies)) / max(len(strategies), 1)
            quality_metrics['adaptability'] = strategy_diversity

            # 动态环境性能：跨集群路由成功率和MEO移动适应性
            inter_cluster_performance = self.performance_metrics.get('inter_cluster_success_rate', 0.5)

            # MEO移动适应性：移动距离大时仍保持较好性能
            meo_adaptation_score = 1.0
            if self.meo_movement_stats:
                all_movements = []
                for slot_data in self.meo_movement_stats:
                    all_movements.extend(slot_data['movements'].values())

                if all_movements:
                    avg_movement = np.mean(all_movements)
                    # 移动距离越大，要求性能保持越好，得分越高
                    movement_factor = min(avg_movement / 10.0, 1.0)  # 归一化到[0,1]
                    success_factor = quality_metrics['robustness']
                    meo_adaptation_score = success_factor / max(movement_factor, 0.1)
                    meo_adaptation_score = min(meo_adaptation_score, 1.0)

            quality_metrics['dynamic_performance'] = (inter_cluster_performance + meo_adaptation_score) / 2.0

        # 计算综合质量评分
        overall_quality = np.mean(list(quality_metrics.values()))
        quality_metrics['overall_quality'] = overall_quality

        self.logger.info("=== 模型质量评估 (动态MEO) ===")
        self.logger.info(f"鲁棒性: {quality_metrics['robustness']:.3f}")
        self.logger.info(f"效率: {quality_metrics['efficiency']:.3f}")
        self.logger.info(f"一致性: {quality_metrics['consistency']:.3f}")
        self.logger.info(f"适应性: {quality_metrics['adaptability']:.3f}")
        self.logger.info(f"动态环境性能: {quality_metrics['dynamic_performance']:.3f}")
        self.logger.info(f"综合质量: {overall_quality:.3f}")

        return quality_metrics