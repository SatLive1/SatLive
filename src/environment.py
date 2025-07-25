from typing import Dict, Tuple, List
import math

from satellites import LEOSatellite, MEOSatellite


def distance(pos1: Tuple[float, float, float], pos2: Tuple[float, float, float]) -> float:
    """Compute Euclidean distance between two 3D points."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))


def distance_2d(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    """Compute 2D Euclidean distance between two points."""
    return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)


def find_nearest_available_leo(
        ground_pos: Tuple[float, float, float],
        leos: Dict[int, LEOSatellite],
        load_threshold: int,
) -> LEOSatellite:
    """Return the nearest LEO with load below threshold."""
    candidates = [leo for leo in leos.values() if leo.load < load_threshold]
    if not candidates:
        raise ValueError("No available LEO satellite")
    return min(candidates, key=lambda l: distance(ground_pos, (l.latitude, l.longitude, l.altitude)))


def find_nearest_meo_to_leo(
        leo: LEOSatellite,
        meos: Dict[int, MEOSatellite]
) -> MEOSatellite:
    """
    找到距离指定LEO最近的MEO卫星

    Args:
        leo: LEO卫星
        meos: MEO卫星字典

    Returns:
        最近的MEO卫星
    """
    if not meos:
        raise ValueError("No MEO satellites available")

    leo_pos = (leo.latitude, leo.longitude, leo.altitude)
    return min(meos.values(),
               key=lambda m: distance(leo_pos, (m.latitude, m.longitude, m.altitude)))


def find_optimal_meo_for_inter_cluster_routing(
        src_leo: LEOSatellite,
        dst_leo: LEOSatellite,
        meos: Dict[int, MEOSatellite],
        distance_weight: float = 0.7,
        load_weight: float = 0.3
) -> MEOSatellite:
    """
    为跨集群路由找到最优的中继MEO卫星

    Args:
        src_leo: 源LEO卫星
        dst_leo: 目标LEO卫星
        meos: MEO卫星字典
        distance_weight: 距离权重
        load_weight: 负载权重

    Returns:
        最优的MEO卫星
    """
    if not meos:
        raise ValueError("No MEO satellites available")

    src_pos = (src_leo.latitude, src_leo.longitude, src_leo.altitude)
    dst_pos = (dst_leo.latitude, dst_leo.longitude, dst_leo.altitude)

    best_meo = None
    best_score = float('inf')

    for meo in meos.values():
        meo_pos = (meo.latitude, meo.longitude, meo.altitude)

        # 计算MEO到源和目标的总距离
        total_distance = distance(src_pos, meo_pos) + distance(meo_pos, dst_pos)

        # 计算MEO的集群负载（所管理的LEO数量）
        cluster_load = len(meo.cluster_leos)

        # 归一化处理
        normalized_distance = total_distance
        normalized_load = cluster_load

        # 计算综合得分
        score = distance_weight * normalized_distance + load_weight * normalized_load

        if score < best_score:
            best_score = score
            best_meo = meo

    return best_meo


def calculate_meo_cluster_connectivity(
        meo: MEOSatellite,
        leos: Dict[int, LEOSatellite]
) -> Dict[str, float]:
    """
    计算MEO集群的连通性指标

    Args:
        meo: MEO卫星
        leos: LEO卫星字典

    Returns:
        连通性指标字典
    """
    if not meo.cluster_leos:
        return {
            'internal_connectivity': 0.0,
            'external_connectivity': 0.0,
            'average_degree': 0.0,
            'cluster_efficiency': 0.0
        }

    cluster_leo_set = set(meo.cluster_leos)

    # 计算集群内连通性
    internal_edges = 0
    external_edges = 0
    total_degree = 0

    for leo_id in meo.cluster_leos:
        if leo_id not in leos:
            continue

        leo = leos[leo_id]
        total_degree += len(leo.neighbors)

        for neighbor_id in leo.neighbors:
            if neighbor_id in cluster_leo_set:
                internal_edges += 1
            else:
                external_edges += 1

    # 避免重复计算内部边
    internal_edges = internal_edges // 2

    # 计算指标
    cluster_size = len(meo.cluster_leos)
    max_internal_edges = cluster_size * (cluster_size - 1) // 2

    internal_connectivity = internal_edges / max_internal_edges if max_internal_edges > 0 else 0.0
    external_connectivity = external_edges / len(meo.cluster_leos) if meo.cluster_leos else 0.0
    average_degree = total_degree / len(meo.cluster_leos) if meo.cluster_leos else 0.0

    # 集群效率：内部连通性与外部连通性的平衡
    cluster_efficiency = internal_connectivity * 0.7 + min(external_connectivity / 4.0, 1.0) * 0.3

    return {
        'internal_connectivity': internal_connectivity,
        'external_connectivity': external_connectivity,
        'average_degree': average_degree,
        'cluster_efficiency': cluster_efficiency,
        'internal_edges': internal_edges,
        'external_edges': external_edges
    }


def update_dynamic_meo_clusters(
        leos: Dict[int, LEOSatellite],
        meos: Dict[int, MEOSatellite],
        reassignment_threshold: float = 0.3
) -> Dict[int, int]:
    """
    基于当前网络状态动态更新MEO集群分配

    Args:
        leos: LEO卫星字典
        meos: MEO卫星字典
        reassignment_threshold: 重新分配的阈值

    Returns:
        更新后的LEO到MEO的分配映射
    """
    # 计算每个LEO到每个MEO的适应度
    leo_meo_fitness = {}

    for leo_id, leo in leos.items():
        leo_meo_fitness[leo_id] = {}
        leo_pos = (leo.latitude, leo.longitude, leo.altitude)

        for meo_id, meo in meos.items():
            meo_pos = (meo.latitude, meo.longitude, meo.altitude)

            # 计算距离适应度（距离越近越好）
            dist = distance(leo_pos, meo_pos)
            distance_fitness = 1.0 / (1.0 + dist / 1000.0)  # 归一化

            # 计算负载适应度（集群越小越好）
            current_cluster_size = len(meo.cluster_leos)
            load_fitness = 1.0 / (1.0 + current_cluster_size / 10.0)  # 归一化

            # 综合适应度
            total_fitness = 0.6 * distance_fitness + 0.4 * load_fitness
            leo_meo_fitness[leo_id][meo_id] = total_fitness

    # 记录原始分配
    original_assignments = {leo.id: leo.meo_id for leo in leos.values()}
    new_assignments = {}

    # 为每个LEO找到最优MEO
    for leo_id, leo in leos.items():
        current_meo_id = leo.meo_id
        current_fitness = leo_meo_fitness[leo_id].get(current_meo_id, 0.0)

        best_meo_id = max(leo_meo_fitness[leo_id].keys(),
                          key=lambda mid: leo_meo_fitness[leo_id][mid])
        best_fitness = leo_meo_fitness[leo_id][best_meo_id]

        # 只有当新分配显著更好时才进行更改
        if best_fitness > current_fitness + reassignment_threshold:
            new_assignments[leo_id] = best_meo_id
        else:
            new_assignments[leo_id] = current_meo_id

    # 更新LEO的MEO分配
    for leo_id, new_meo_id in new_assignments.items():
        leos[leo_id].meo_id = new_meo_id

    # 重新构建MEO的cluster_leos列表
    for meo in meos.values():
        meo.cluster_leos = []

    for leo_id, meo_id in new_assignments.items():
        if meo_id in meos:
            meos[meo_id].cluster_leos.append(leo_id)

    return new_assignments


def analyze_network_topology(
        leos: Dict[int, LEOSatellite],
        meos: Dict[int, MEOSatellite]
) -> Dict[str, any]:
    """
    分析整个网络拓扑的特征

    Args:
        leos: LEO卫星字典
        meos: MEO卫星字典

    Returns:
        网络拓扑分析结果
    """
    analysis = {
        'total_leos': len(leos),
        'total_meos': len(meos),
        'total_edges': 0,
        'average_degree': 0.0,
        'cluster_analysis': {},
        'network_efficiency': 0.0,
        'meo_positions': [],
        'leo_meo_distances': []
    }

    # 统计边数和度数
    total_degree = 0
    for leo in leos.values():
        total_degree += len(leo.neighbors)

    analysis['total_edges'] = total_degree // 2  # 无向图，避免重复计算
    analysis['average_degree'] = total_degree / len(leos) if leos else 0.0

    # 分析每个MEO集群
    for meo_id, meo in meos.items():
        cluster_stats = calculate_meo_cluster_connectivity(meo, leos)
        analysis['cluster_analysis'][meo_id] = cluster_stats

        # 记录MEO位置
        analysis['meo_positions'].append({
            'meo_id': meo_id,
            'latitude': meo.latitude,
            'longitude': meo.longitude,
            'altitude': meo.altitude,
            'cluster_size': len(meo.cluster_leos)
        })

    # 计算LEO到其管理MEO的距离分布
    for leo in leos.values():
        if leo.meo_id in meos:
            meo = meos[leo.meo_id]
            leo_pos = (leo.latitude, leo.longitude, leo.altitude)
            meo_pos = (meo.latitude, meo.longitude, meo.altitude)
            dist = distance(leo_pos, meo_pos)
            analysis['leo_meo_distances'].append(dist)

    # 计算网络整体效率
    if analysis['cluster_analysis']:
        cluster_efficiencies = [stats['cluster_efficiency']
                                for stats in analysis['cluster_analysis'].values()]
        analysis['network_efficiency'] = sum(cluster_efficiencies) / len(cluster_efficiencies)

    return analysis


def get_leo_by_id(leos: Dict[int, LEOSatellite], leo_id: int) -> LEOSatellite:
    if leo_id not in leos:
        raise ValueError(f"LEO {leo_id} not found")
    return leos[leo_id]


def get_meo_by_id(meos: Dict[int, MEOSatellite], meo_id: int) -> MEOSatellite:
    if meo_id not in meos:
        raise ValueError(f"MEO {meo_id} not found")
    return meos[meo_id]


def get_meo_by_position(
        meos: Dict[int, MEOSatellite],
        target_position: Tuple[float, float, float],
        max_distance: float = 100.0
) -> MEOSatellite:
    """
    根据位置查找MEO卫星

    Args:
        meos: MEO卫星字典
        target_position: 目标位置 (lat, lon, alt)
        max_distance: 最大匹配距离

    Returns:
        最近的MEO卫星
    """
    best_meo = None
    best_distance = float('inf')

    for meo in meos.values():
        meo_pos = (meo.latitude, meo.longitude, meo.altitude)
        dist = distance(target_position, meo_pos)

        if dist < best_distance and dist <= max_distance:
            best_distance = dist
            best_meo = meo

    if best_meo is None:
        raise ValueError(f"No MEO found within {max_distance} units of position {target_position}")

    return best_meo