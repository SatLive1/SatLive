"""
基于data.json的数据加载和环境初始化模块
包含动态MEO卫星信息和LEO-MEO分配关系
"""

import json
from typing import Dict, Tuple, List
from src.satellites import LEOSatellite, MEOSatellite

def load_environment_from_json(json_file: str = "data.json") -> Tuple[Dict[int, LEOSatellite], Dict[int, MEOSatellite], dict]:
    """
    从JSON文件加载完整的卫星环境数据

    Args:
        json_file: JSON数据文件路径

    Returns:
        (leos, meos, raw_data): LEO卫星字典, MEO卫星字典, 原始数据
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    # 创建空的卫星字典，具体位置需要在特定时间槽时加载
    leos = {}
    meos = {}

    return leos, meos, data

def get_leo_meo_assignment(slot_id: int, data: dict) -> List[int]:
    """
    获取指定时间槽的LEO-MEO分配关系

    Args:
        slot_id: 时间槽ID
        data: 原始JSON数据

    Returns:
        LEO卫星对应的MEO控制节点ID列表
    """
    meo_assignments = data.get('MEO_per_slot', [])
    for slot_info in meo_assignments:
        if slot_info['slot_id'] == slot_id:
            return slot_info['leo_meo_assignments']

    # 如果没找到，返回默认分配
    num_leos = data.get('num_satellites', 10)
    num_meos = data.get('num_meo_satellites', 3)
    return [i % num_meos for i in range(num_leos)]

def get_meo_positions_for_slot(slot_id: int, data: dict) -> List[List[float]]:
    """
    获取指定时间槽的MEO卫星位置信息

    Args:
        slot_id: 时间槽ID
        data: 原始JSON数据

    Returns:
        MEO卫星位置列表 [[lat, lon, alt], ...]
    """
    # 检查是否有动态MEO位置数据
    if 'meo_positions_per_slot' in data:
        # 新的动态MEO位置格式
        if slot_id < len(data['meo_positions_per_slot']):
            return data['meo_positions_per_slot'][slot_id]

    # 回退到静态MEO位置（向后兼容）
    return data.get('meo_positions', [])

def create_leos_for_slot(slot_id: int, data: dict) -> Dict[int, LEOSatellite]:
    """
    为指定时间槽创建LEO卫星

    Args:
        slot_id: 时间槽ID
        data: 原始JSON数据

    Returns:
        LEO卫星字典
    """
    leos = {}

    # 获取该时间槽的卫星位置
    if slot_id >= len(data['sat_positions_per_slot']):
        raise ValueError(f"时间槽 {slot_id} 超出可用数据范围 (0-{len(data['sat_positions_per_slot'])-1})")

    sat_positions = data['sat_positions_per_slot'][slot_id]

    # 获取该时间槽的邻居关系
    neighbors_info = None
    for neighbor_slot in data['neighbors_per_slot']:
        if neighbor_slot['slot_id'] == slot_id:
            neighbors_info = neighbor_slot['neighbors']
            break

    # 获取MEO分配
    meo_assignments = get_leo_meo_assignment(slot_id, data)

    # 创建LEO卫星
    for i, (lat, lon) in enumerate(sat_positions):
        neighbors = neighbors_info[i] if neighbors_info else []
        meo_id = meo_assignments[i] if i < len(meo_assignments) else 0

        leos[i] = LEOSatellite(
            id=i,
            latitude=lat,
            longitude=lon,
            altitude=500.0,  # 默认LEO高度
            load=0,  # 初始负载为0
            neighbors=neighbors,
            meo_id=meo_id
        )

    return leos

def create_meos_for_slot(slot_id: int, data: dict) -> Dict[int, MEOSatellite]:
    """
    为指定时间槽创建MEO卫星

    Args:
        slot_id: 时间槽ID
        data: 原始JSON数据

    Returns:
        MEO卫星字典
    """
    meos = {}

    # 获取该时间槽的MEO位置
    meo_positions = get_meo_positions_for_slot(slot_id, data)

    # 创建MEO卫星
    for i, position in enumerate(meo_positions):
        if len(position) >= 3:
            lat, lon, alt = position[0], position[1], position[2]
        else:
            # 兼容只有2D位置的情况
            lat, lon = position[0], position[1]
            alt = 1000.0  # 默认MEO高度

        meos[i] = MEOSatellite(
            id=i,
            latitude=lat,
            longitude=lon,
            altitude=alt,
            cluster_leos=[]  # 将在后面动态分配
        )

    return meos

def update_meo_clusters(leos: Dict[int, LEOSatellite], meos: Dict[int, MEOSatellite]):
    """
    根据LEO的MEO分配更新MEO的cluster_leos列表

    Args:
        leos: LEO卫星字典
        meos: MEO卫星字典
    """
    # 清空所有MEO的cluster列表
    for meo in meos.values():
        meo.cluster_leos = []

    # 根据LEO的meo_id重新分配
    for leo in leos.values():
        if leo.meo_id in meos:
            meos[leo.meo_id].cluster_leos.append(leo.id)

def load_complete_environment(slot_id: int, json_file: str = "data.json") -> Tuple[Dict[int, LEOSatellite], Dict[int, MEOSatellite], dict]:
    """
    加载指定时间槽的完整环境（支持动态MEO）

    Args:
        slot_id: 时间槽ID
        json_file: JSON数据文件路径

    Returns:
        (leos, meos, raw_data): 完整的环境数据
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    # 创建LEO卫星
    leos = create_leos_for_slot(slot_id, data)

    # 创建MEO卫星（动态位置）
    meos = create_meos_for_slot(slot_id, data)

    # 更新MEO的cluster信息
    update_meo_clusters(leos, meos)

    return leos, meos, data

def validate_dynamic_meo_data(data: dict) -> bool:
    """
    验证动态MEO数据的完整性

    Args:
        data: 原始JSON数据

    Returns:
        数据是否有效
    """
    # 检查是否有动态MEO位置数据
    if 'meo_positions_per_slot' not in data:
        print("警告: 没有找到动态MEO位置数据 (meo_positions_per_slot)，将使用静态MEO位置")
        return False

    meo_positions_per_slot = data['meo_positions_per_slot']
    sat_positions_per_slot = data.get('sat_positions_per_slot', [])

    # 检查时间槽数量是否匹配
    if len(meo_positions_per_slot) != len(sat_positions_per_slot):
        print(f"警告: MEO位置时间槽数 ({len(meo_positions_per_slot)}) 与LEO位置时间槽数 ({len(sat_positions_per_slot)}) 不匹配")
        return False

    # 检查每个时间槽的MEO数量是否一致
    num_meos = data.get('num_meo_satellites', 3)
    for slot_id, meo_positions in enumerate(meo_positions_per_slot):
        if len(meo_positions) != num_meos:
            print(f"警告: 时间槽 {slot_id} 的MEO数量 ({len(meo_positions)}) 与配置不符 ({num_meos})")
            return False

    print(f"动态MEO数据验证成功: {len(meo_positions_per_slot)} 个时间槽，每个时间槽 {num_meos} 个MEO")
    return True

def print_environment_summary(leos: Dict[int, LEOSatellite], meos: Dict[int, MEOSatellite], slot_id: int):
    """打印环境摘要信息（增强版，包含MEO位置信息）"""
    print(f"\n=== 时间槽 {slot_id} 环境摘要 ===")
    print(f"LEO卫星数量: {len(leos)}")
    print(f"MEO卫星数量: {len(meos)}")

    # 显示MEO位置信息
    print("\nMEO卫星位置:")
    for meo_id, meo in meos.items():
        print(f"  MEO {meo_id}: 位置({meo.latitude:.1f}, {meo.longitude:.1f}, {meo.altitude:.1f})")

    # 统计各MEO的cluster大小
    print("\nMEO集群分配:")
    for meo_id, meo in meos.items():
        print(f"  MEO {meo_id}: 控制 {len(meo.cluster_leos)} 个LEO卫星 {meo.cluster_leos}")

    # 显示前几个LEO的信息
    print("\n前5个LEO卫星信息:")
    for i in range(min(5, len(leos))):
        leo = leos[i]
        print(f"  LEO {i}: 位置({leo.latitude:.1f}, {leo.longitude:.1f}), 控制MEO={leo.meo_id}, 邻居={leo.neighbors}")

def generate_sample_dynamic_meo_data(num_slots: int, num_meos: int) -> List[List[List[float]]]:
    """
    生成示例动态MEO数据（用于测试）

    Args:
        num_slots: 时间槽数量
        num_meos: MEO卫星数量

    Returns:
        动态MEO位置数据 [slot][meo_id][lat, lon, alt]
    """
    import random

    meo_positions_per_slot = []

    # 为每个MEO设置初始位置
    base_positions = []
    for i in range(num_meos):
        base_lat = 45.0 + i * 10.0  # 分散在不同纬度
        base_lon = 45.0 + i * 10.0  # 分散在不同经度
        base_alt = 1000.0  # MEO高度
        base_positions.append([base_lat, base_lon, base_alt])

    # 为每个时间槽生成MEO位置（在基础位置附近移动）
    for slot in range(num_slots):
        slot_positions = []
        for meo_id in range(num_meos):
            base_lat, base_lon, base_alt = base_positions[meo_id]

            # 在基础位置附近随机移动（模拟轨道运动）
            movement_range = 5.0  # 移动范围
            new_lat = base_lat + random.uniform(-movement_range, movement_range)
            new_lon = base_lon + random.uniform(-movement_range, movement_range)
            new_alt = base_alt + random.uniform(-50.0, 50.0)  # 高度也有小幅变化

            slot_positions.append([new_lat, new_lon, new_alt])

        meo_positions_per_slot.append(slot_positions)

    return meo_positions_per_slot

if __name__ == "__main__":
    # 测试动态MEO数据加载

    # 首先尝试加载现有数据并验证
    try:
        with open('data/data.json', 'r') as f:
            data = json.load(f)

        is_valid = validate_dynamic_meo_data(data)

        if not is_valid:
            print("\n生成示例动态MEO数据...")
            # 生成示例动态MEO数据
            num_slots = len(data.get('sat_positions_per_slot', []))
            num_meos = data.get('num_meo_satellites', 3)

            if num_slots > 0:
                sample_meo_data = generate_sample_dynamic_meo_data(num_slots, num_meos)
                print(f"生成了 {num_slots} 个时间槽的动态MEO数据")
                print("示例数据 (前3个时间槽):")
                for slot_id in range(min(3, len(sample_meo_data))):
                    print(f"  时间槽 {slot_id}: {sample_meo_data[slot_id]}")

        # 测试几个时间槽的数据加载
        print("\n测试动态MEO环境加载:")
        for slot_id in [0, 25, 49]:
            try:
                leos, meos, data = load_complete_environment(slot_id)
                print_environment_summary(leos, meos, slot_id)
            except Exception as e:
                print(f"加载时间槽 {slot_id} 失败: {e}")

    except FileNotFoundError:
        print("数据文件不存在，请确保 data/data.json 文件存在")
    except Exception as e:
        print(f"测试失败: {e}")