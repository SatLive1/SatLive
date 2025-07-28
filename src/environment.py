from typing import Dict, Tuple, List
import math

try:
    from satellites import LEOSatellite, MEOSatellite
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保在正确的目录下运行脚本")
    import sys
    sys.exit(1)


def distance(pos1: Tuple[float, float, float], pos2: Tuple[float, float, float]) -> float:
    """Compute Euclidean distance between two 3D points."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))


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
        'internal_connectivity': internal_connectivity, # 集群内部连通密度
        'external_connectivity': external_connectivity, # 集群对外连接能力
        'average_degree': average_degree, # 平均节点度数
        'cluster_efficiency': cluster_efficiency, # 综合效率评估
        'internal_edges': internal_edges,
        'external_edges': external_edges
    }

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
