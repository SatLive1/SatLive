"""MEO-LEO集群路由训练脚本 - 支持动态MEO"""
import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)
import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import logging
from datetime import datetime

try:
    from config import Config
    from satellites import LEOSatellite, MEOSatellite
    from rl_agent import RLAgent
    from routing import route_request_with_intelligent_edge_selection
    from environment import find_nearest_available_leo, analyze_network_topology, update_dynamic_meo_clusters
    from data.data_loader import load_complete_environment, validate_dynamic_meo_data
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保在正确的目录下运行脚本")
    sys.exit(1)


class TrainingEnvironment:
    """训练环境类 - 支持动态MEO"""

    def __init__(self, config: Config):
        self.config = config
        self.setup_logging()
        self.setup_directories()

        # 训练统计
        self.episode_rewards = []
        self.episode_success_rates = []
        self.episode_avg_path_lengths = []

        # 动态MEO相关统计
        self.meo_position_stats = []
        self.cluster_reassignment_stats = []
        self.network_topology_stats = []

    def setup_logging(self):
        """设置日志"""
        log_level = getattr(logging, self.config.get('output.log_level', 'INFO'))
        log_file = self.config.get('output.log_file', 'logs/training.log')

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

    def setup_directories(self):
        """创建必要的目录"""
        directories = [
            self.config.get('output.model_save_path', 'models/'),
            self.config.get('output.results_path', 'results/'),
            'logs/'
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def validate_data_compatibility(self, data_file: str) -> bool:
        """验证数据文件是否支持动态MEO"""
        try:
            with open(data_file, 'r') as f:
                data = json.load(f)

            is_dynamic_meo = validate_dynamic_meo_data(data)

            if is_dynamic_meo:
                self.logger.info("检测到动态MEO数据，启用动态MEO训练模式")
            else:
                self.logger.info("使用静态MEO数据，MEO位置在所有时间槽保持不变")

            return True

        except Exception as e:
            self.logger.error(f"数据验证失败: {e}")
            return False

    def train(self):
        """执行训练"""
        self.logger.info("开始训练...")

        # 设置随机种子
        random_seed = self.config.get('simulation.random_seed', 42)
        random.seed(random_seed)
        np.random.seed(random_seed)

        # 获取数据文件路径并验证
        data_file = self.config.get('data.data_file', 'data/data.json')
        if not self.validate_data_compatibility(data_file):
            self.logger.error("数据文件验证失败，无法开始训练")
            return

        # 初始化智能体
        agent = RLAgent(
            learning_rate=self.config.get('rl_agent.learning_rate', 0.1),
            gamma=self.config.get('rl_agent.gamma', 0.9),
            epsilon=self.config.get('rl_agent.epsilon', 0.1)
        )

        # 训练参数
        num_episodes = self.config.get('training.num_episodes', 1000)
        max_steps = self.config.get('training.max_steps_per_episode', 100)
        save_interval = self.config.get('training.save_interval', 100)

        for episode in range(num_episodes):
            episode_reward, success_rate, avg_path_length = self.run_episode(
                agent, data_file, max_steps
            )

            # 记录统计信息
            self.episode_rewards.append(episode_reward)
            self.episode_success_rates.append(success_rate)
            self.episode_avg_path_lengths.append(avg_path_length)

            # 更新epsilon（探索率衰减）
            epsilon_decay = self.config.get('rl_agent.epsilon_decay', 0.995)
            epsilon_min = self.config.get('rl_agent.epsilon_min', 0.01)
            agent.epsilon = max(epsilon_min, agent.epsilon * epsilon_decay)

            # 日志输出
            if episode % 100 == 0:
                self.logger.info(
                    f"Episode {episode}: Reward={episode_reward:.2f}, "
                    f"Success Rate={success_rate:.2f}, "
                    f"Avg Path Length={avg_path_length:.2f}, "
                    f"Epsilon={agent.epsilon:.3f}"
                )

            # 保存模型
            if episode % save_interval == 0 and episode > 0:
                self.save_model(agent, episode)

        # 训练完成
        self.logger.info("训练完成!")
        self.save_final_results(agent)

        if self.config.get('output.plot_results', True):
            self.plot_training_results()

    def run_episode(self, agent: RLAgent, data_file: str, max_steps: int) -> Tuple[float, float, float]:
        """改进的episode运行逻辑 - 支持动态MEO"""

        # 载入训练查询和时间槽数量
        with open(data_file, 'r') as f:
            data = json.load(f)
        train_queries = data.get('train_queries', [])
        num_time_slots = data.get('num_train_slots', self.config.get('network.num_time_slots', 50))

        # 统计数据
        total_reward = 0.0
        successful_routes = 0
        total_routes = 0
        total_path_length = 0

        # 包管理：每个包的状态 {packet_id: PacketInfo}
        active_packets = {}
        completed_packets = []
        next_packet_id = 0

        # 动态MEO统计
        meo_reassignments = 0
        topology_changes = []

        # 按slot进行模拟
        for current_slot in range(num_time_slots):
            if total_routes >= max_steps:
                break

            self.logger.debug(f"\n=== Processing Slot {current_slot} ===")

            # 1. 处理新到达的查询请求
            new_queries = [q for q in train_queries if q['time'] == current_slot]
            for query in new_queries:
                packet_info = {
                    'id': next_packet_id,
                    'src': query['src'],
                    'dst': query['dst'],
                    'current_pos': query['src'],
                    'path': [],
                    'path_index': 0,  # 当前在路径中的位置
                    'start_slot': current_slot,
                    'hops_completed': 0,
                    'status': 'new'  # new, routing, forwarding, completed, failed
                }
                active_packets[next_packet_id] = packet_info
                total_routes += 1
                next_packet_id += 1
                self.logger.debug(f"New packet {next_packet_id - 1} arrived: {query['src']} -> {query['dst']}")

            # 2. 载入当前slot的网络环境（包含动态MEO）
            try:
                leos, meos, _ = load_complete_environment(current_slot, data_file, neighbors_dir="data/neighbors")
            except Exception as e:
                self.logger.error(f"加载时间槽 {current_slot} 环境失败: {e}")
                continue

            # 3. 分析网络拓扑变化（可选）
            if current_slot % 10 == 0:  # 每10个slot分析一次
                topology_analysis = analyze_network_topology(leos, meos)
                topology_changes.append({
                    'slot': current_slot,
                    'analysis': topology_analysis
                })

                self.logger.debug(f"Slot {current_slot} topology: "
                                f"Network efficiency = {topology_analysis['network_efficiency']:.3f}")

            # 4. 动态MEO集群重分配（可选，用于适应MEO移动）
            enable_dynamic_reassignment = self.config.get('network.enable_dynamic_meo_reassignment', False)
            if enable_dynamic_reassignment:  # 每5个slot重分配一次
                try:
                    original_assignments = {leo.id: leo.meo_id for leo in leos.values()}
                    new_assignments = update_dynamic_meo_clusters(leos, meos)

                    # 统计重分配情况
                    reassignment_count = sum(1 for leo_id in original_assignments
                                           if original_assignments[leo_id] != new_assignments.get(leo_id, -1))

                    if reassignment_count > 0:
                        meo_reassignments += reassignment_count
                        self.logger.debug(f"Slot {current_slot}: {reassignment_count} LEOs reassigned to different MEOs")

                except Exception as e:
                    self.logger.warning(f"Dynamic MEO reassignment failed at slot {current_slot}: {e}")

            # 5. 处理所有活跃的包
            packets_to_remove = []
            for packet_id, packet in active_packets.items():
                reward = 0.0

                if packet['status'] == 'new':
                    # 新包需要进行路由决策
                    reward = self.process_packet_routing(packet, agent, leos, meos, current_slot)

                elif packet['status'] == 'forwarding':
                    # 在转发中的包，每个slot只转发一跳
                    reward = self.process_packet_single_hop(packet, leos, current_slot)

                total_reward += reward

                # 检查包是否完成或失败
                if packet['status'] in ['completed', 'failed']:
                    packets_to_remove.append(packet_id)
                    completed_packets.append(packet.copy())

                    if packet['status'] == 'completed':
                        successful_routes += 1
                        total_path_length += packet['hops_completed']

            # 6. 移除完成的包
            for packet_id in packets_to_remove:
                del active_packets[packet_id]

            # 日志输出
            if current_slot % 10 == 0:
                active_count = len(active_packets)
                new_count = len(new_queries)
                if new_count > 0 or active_count > 0:
                    self.logger.debug(f"Slot {current_slot}: New={new_count}, Active={active_count}")

        # 处理仍在转发中的包（超时）
        for packet in active_packets.values():
            packet['status'] = 'failed'
            packet['end_slot'] = num_time_slots - 1
            completed_packets.append(packet.copy())
            timeout_penalty = self.config.get('environment.reward_timeout', -2.0)
            total_reward += timeout_penalty

        # 记录动态MEO统计信息
        if meo_reassignments > 0:
            self.cluster_reassignment_stats.append(meo_reassignments)

        if topology_changes:
            self.network_topology_stats.extend(topology_changes)

        # 计算统计数据
        success_rate = successful_routes / total_routes if total_routes > 0 else 0.0
        avg_path_length = total_path_length / successful_routes if successful_routes > 0 else 0.0

        if meo_reassignments > 0:
            self.logger.info(
                f"Episode completed: {successful_routes}/{total_routes} successful, "
                f"avg_path_length: {avg_path_length:.2f}, MEO reassignments: {meo_reassignments}")
        else:
            self.logger.info(
                f"Episode completed: {successful_routes}/{total_routes} successful, "
                f"avg_path_length: {avg_path_length:.2f}")

        return total_reward, success_rate, avg_path_length

    def process_packet_routing(self, packet: Dict, agent: RLAgent,
                               leos: Dict[int, LEOSatellite],
                               meos: Dict[int, MEOSatellite],
                               current_slot: int) -> float:
        """处理包的路由决策阶段 - 支持动态MEO"""

        src_id = packet['current_pos']
        dst_id = packet['dst']

        # 验证源和目标LEO是否存在于当前网络环境中
        if src_id not in leos or dst_id not in leos:
            packet['status'] = 'failed'
            packet['end_slot'] = current_slot
            failure_reward = self.config.get('environment.reward_failure', -5.0)
            self.logger.debug(f"Packet {packet['id']}: Source or destination LEO not available")
            return failure_reward

        # 使用智能代理进行路由决策（考虑动态MEO位置）
        try:
            path, routing_stats = route_request_with_intelligent_edge_selection(
                src_id, dst_id, leos, meos, agent
            )
        except Exception as e:
            self.logger.debug(f"Packet {packet['id']}: Routing error - {e}")
            packet['status'] = 'failed'
            packet['end_slot'] = current_slot
            failure_reward = self.config.get('environment.reward_failure', -5.0)
            return failure_reward

        if not path or len(path) < 2:
            # 路由失败
            packet['status'] = 'failed'
            packet['end_slot'] = current_slot
            failure_reward = self.config.get('environment.reward_failure', -5.0)
            self.logger.debug(f"Packet {packet['id']}: Routing failed from {src_id} to {dst_id}")
            return failure_reward

        # 验证路径有效性
        if not self.validate_path(path, leos):
            packet['status'] = 'failed'
            packet['end_slot'] = current_slot
            invalid_reward = self.config.get('environment.reward_failure', -5.0)
            self.logger.debug(f"Packet {packet['id']}: Invalid path")
            return invalid_reward

        # 路由成功，设置包的路径
        packet['path'] = path
        packet['path_index'] = 0  # 从路径起点开始
        packet['routing_stats'] = routing_stats  # 保存路由统计信息

        # 如果源和目标相同，直接完成
        if src_id == dst_id:
            packet['status'] = 'completed'
            packet['end_slot'] = current_slot
            packet['hops_completed'] = 0
            success_reward = self.config.get('environment.reward_success', 10.0)
            self.logger.debug(f"Packet {packet['id']}: Same src/dst, completed immediately")
            return success_reward

        # 开始转发流程
        packet['status'] = 'forwarding'

        # 给予路由成功奖励，考虑路由策略
        routing_reward = self.config.get('environment.reward_routing_success', 1.0)

        # 如果使用了跨集群路由，给予额外奖励
        if routing_stats.get('routing_strategy') == 'inter_cluster_two_stage':
            routing_reward += 0.5  # 跨集群路由额外奖励

        self.logger.debug(f"Packet {packet['id']}: Route found with {len(path)} hops, "
                         f"strategy: {routing_stats.get('routing_strategy', 'unknown')}")
        return routing_reward

    def process_packet_single_hop(self, packet: Dict,
                                  leos: Dict[int, LEOSatellite],
                                  current_slot: int) -> float:
        """处理包的单跳转发 - 适应动态网络拓扑"""

        # 检查是否已经到达路径末尾
        if packet['path_index'] >= len(packet['path']) - 1:
            # 已经在目标节点，完成传输
            packet['status'] = 'completed'
            packet['end_slot'] = current_slot
            completion_reward = self.calculate_completion_reward(packet, leos)
            self.logger.debug(f"Packet {packet['id']}: Completed transmission at slot {current_slot}")
            return completion_reward

        # 获取当前节点和下一跳节点
        current_node = packet['path'][packet['path_index']]
        next_node = packet['path'][packet['path_index'] + 1]

        # 检查当前节点是否仍存在（动态网络拓扑）
        if current_node not in leos:
            packet['status'] = 'failed'
            packet['end_slot'] = current_slot
            node_lost_penalty = self.config.get('environment.reward_connection_lost', -1.0)
            self.logger.debug(f"Packet {packet['id']}: Current node {current_node} no longer exists")
            return node_lost_penalty

        # 检查连接是否仍然存在（网络拓扑可能发生变化）
        if next_node not in leos or next_node not in leos[current_node].neighbors:
            # 连接断开，需要重新路由
            packet['status'] = 'new'  # 重新进入路由阶段
            packet['current_pos'] = current_node
            packet['path'] = []  # 清空旧路径
            packet['path_index'] = 0

            connection_lost_penalty = self.config.get('environment.reward_connection_lost', -1.0)
            self.logger.debug(f"Packet {packet['id']}: Connection lost from {current_node} to {next_node}, re-routing")
            return connection_lost_penalty

        # 执行单跳转发
        packet['path_index'] += 1
        packet['current_pos'] = next_node
        packet['hops_completed'] += 1

        # 更新目标卫星的负载
        if next_node in leos:
            leos[next_node].load += 1

        # 检查是否到达目的地
        if next_node == packet['dst']:
            packet['status'] = 'completed'
            packet['end_slot'] = current_slot

            # 计算完成奖励
            completion_reward = self.calculate_completion_reward(packet, leos)
            self.logger.debug(
                f"Packet {packet['id']}: Reached destination at slot {current_slot} after {packet['hops_completed']} hops")
            return completion_reward

        # 还在转发中，给予转发奖励
        forwarding_reward = self.config.get('environment.reward_forwarding', 0.1)
        self.logger.debug(f"Packet {packet['id']}: Forwarded to node {next_node} (hop {packet['hops_completed']})")
        return forwarding_reward

    def calculate_completion_reward(self, packet: Dict, leos: Dict[int, LEOSatellite]) -> float:
        """计算包完成传输时的奖励 - 考虑动态MEO因素"""

        # 基础完成奖励
        reward = self.config.get('environment.reward_success', 10.0)

        # 跳数效率奖励/惩罚
        hop_penalty = self.config.get('environment.reward_hop', -0.1)
        reward += hop_penalty * packet['hops_completed']

        # 传输时延奖励/惩罚
        if 'end_slot' in packet and 'start_slot' in packet:
            transmission_delay = packet['end_slot'] - packet['start_slot']
            delay_penalty = self.config.get('environment.reward_delay', -0.2)
            reward += delay_penalty * transmission_delay

            # 时延效率奖励：如果传输时延接近跳数，说明没有过多的重路由
            if transmission_delay > 0 and packet['hops_completed'] > 0:
                efficiency = packet['hops_completed'] / transmission_delay
                if efficiency > 0.8:  # 效率较高
                    reward += 1.0

        # 负载均衡考虑
        if packet['path'] and len(packet['path']) > 1:
            path_loads = []
            for sat_id in packet['path']:
                if sat_id in leos:
                    path_loads.append(leos[sat_id].load)

            if path_loads:
                avg_load = sum(path_loads) / len(path_loads)
                max_load = self.config.get('environment.max_load_per_satellite', 10)

                # 负载均衡奖励
                load_balance_reward = self.config.get('environment.reward_load_balance', 0.5)
                if avg_load < max_load * 0.5:
                    reward += load_balance_reward
                elif avg_load > max_load * 0.8:
                    reward -= load_balance_reward

        # 路由策略奖励
        if 'routing_stats' in packet:
            routing_strategy = packet['routing_stats'].get('routing_strategy', 'unknown')
            if routing_strategy == 'inter_cluster_two_stage':
                reward += 0.5  # 成功的跨集群路由额外奖励

        return reward

    def validate_path(self, path: List[int], leos: Dict[int, LEOSatellite]) -> bool:
        """验证路径的连通性 - 适应动态网络"""
        if len(path) < 2:
            return True

        for i in range(len(path) - 1):
            current_sat = path[i]
            next_sat = path[i + 1]

            # 检查当前卫星是否存在
            if current_sat not in leos:
                return False

            # 检查连接是否存在
            if next_sat not in leos[current_sat].neighbors:
                return False

        return True

    def generate_random_route_request(self, leos: Dict[int, LEOSatellite]) -> Tuple[int, int]:
        """生成随机路由请求"""
        available_leos = list(leos.keys())
        src_id = random.choice(available_leos)
        dst_id = random.choice(available_leos)
        return src_id, dst_id

    def save_model(self, agent: RLAgent, episode: int):
        """保存模型 - 包含动态MEO训练信息"""
        model_path = self.config.get('output.model_save_path', 'models/')
        model_file = os.path.join(model_path, f'rl_agent_episode_{episode}.json')

        model_data = {
            'episode': episode,
            'q_table': {str(k): v for k, v in agent.q_table.items()},
            'learning_rate': agent.lr,
            'gamma': agent.gamma,
            'epsilon': agent.epsilon,
            'dynamic_meo_enabled': True,  # 标记为动态MEO训练
            'training_stats': {
                'total_reassignments': sum(self.cluster_reassignment_stats),
                'topology_analyses': len(self.network_topology_stats)
            }
        }

        with open(model_file, 'w') as f:
            json.dump(model_data, f, indent=2)

        self.logger.info(f"模型已保存到: {model_file}")

    def save_final_results(self, agent: RLAgent):
        """保存最终结果 - 包含动态MEO统计"""
        results_path = self.config.get('output.results_path', 'results/')

        # 保存最终模型
        final_model_file = os.path.join(results_path, 'final_model.json')
        model_data = {
            'q_table': {str(k): v for k, v in agent.q_table.items()},
            'learning_rate': agent.lr,
            'gamma': agent.gamma,
            'epsilon': agent.epsilon,
            'dynamic_meo_enabled': True,
            'training_completed': datetime.now().isoformat()
        }

        with open(final_model_file, 'w') as f:
            json.dump(model_data, f, indent=2)

        # 保存训练统计
        stats_file = os.path.join(results_path, 'training_stats.json')
        stats = {
            'episode_rewards': self.episode_rewards,
            'episode_success_rates': self.episode_success_rates,
            'episode_avg_path_lengths': self.episode_avg_path_lengths,
            'cluster_reassignment_stats': self.cluster_reassignment_stats,
            'network_topology_stats': self.network_topology_stats,
            'config': self.config.config
        }

        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        self.logger.info(f"最终结果已保存到: {results_path}")

    def plot_training_results(self):
        """绘制训练结果 - 包含动态MEO相关图表"""
        results_path = self.config.get('output.results_path', 'results/')

        # 创建更大的图表布局以容纳额外的统计信息
        fig_size = (20, 10) if self.cluster_reassignment_stats or self.network_topology_stats else (15, 5)
        num_cols = 4 if self.cluster_reassignment_stats or self.network_topology_stats else 3

        plt.figure(figsize=fig_size)

        # 奖励曲线
        plt.subplot(2, num_cols, 1)
        plt.plot(self.episode_rewards)
        plt.title('Training Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)

        # 成功率曲线
        plt.subplot(2, num_cols, 2)
        plt.plot(self.episode_success_rates)
        plt.title('Success Rate')
        plt.xlabel('Episode')
        plt.ylabel('Success Rate')
        plt.grid(True)

        # 平均路径长度
        plt.subplot(2, num_cols, 3)
        plt.plot(self.episode_avg_path_lengths)
        plt.title('Average Path Length')
        plt.xlabel('Episode')
        plt.ylabel('Path Length')
        plt.grid(True)

        # 如果有动态MEO统计，添加额外图表
        if self.cluster_reassignment_stats:
            plt.subplot(2, num_cols, 4)
            plt.plot(self.cluster_reassignment_stats)
            plt.title('MEO Cluster Reassignments')
            plt.xlabel('Episode')
            plt.ylabel('Reassignments')
            plt.grid(True)

        if self.network_topology_stats:
            # 网络效率变化
            plt.subplot(2, num_cols, 5)
            slots = [stat['slot'] for stat in self.network_topology_stats]
            efficiencies = [stat['analysis']['network_efficiency'] for stat in self.network_topology_stats]
            plt.plot(slots, efficiencies)
            plt.title('Network Efficiency Over Time')
            plt.xlabel('Time Slot')
            plt.ylabel('Network Efficiency')
            plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(results_path, 'training_results.png'))
        plt.show()

        self.logger.info(f"训练结果图表已保存到: {results_path}")