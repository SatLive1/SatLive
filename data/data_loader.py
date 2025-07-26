"""
基于分离的JSON文件的数据加载和环境初始化模块
包含动态MEO卫星信息和LEO-MEO分配关系
支持从独立的JSON文件加载：sat_positions_per_slot.json, meo_positions_per_slot.json, MEO_per_slot.json
"""

import json
import os
from typing import Dict, Tuple, List
from src.satellites import LEOSatellite, MEOSatellite

def load_sat_positions_per_slot(file_path: str = "data/sat_positions_per_slot.json") -> List[List[List[float]]]:
    """
    从独立文件加载LEO卫星位置数据

    Args:
        file_path: sat_positions_per_slot.json文件路径

    Returns:
        LEO卫星位置数据 [slot][satellite_id][lat, lon]
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"警告: 文件 {file_path} 不存在，尝试从data.json加载")
        return None

def load_meo_positions_per_slot(file_path: str = "data/meo_positions_per_slot.json") -> List[List[List[float]]]:
    """
    从独立文件加载MEO卫星位置数据

    Args:
        file_path: meo_positions_per_slot.json文件路径

    Returns:
        MEO卫星位置数据 [slot][meo_id][lat, lon, alt]
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"警告: 文件 {file_path} 不存在，尝试从data.json加载")
        return None

def load_meo_per_slot(file_path: str = "data/MEO_per_slot.json") -> List[Dict]:
    """
    从独立文件加载LEO-MEO分配关系数据

    Args:
        file_path: MEO_per_slot.json文件路径

    Returns:
        LEO-MEO分配数据 [{"slot_id": int, "leo_meo_assignments": [int]}]
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"警告: 文件 {file_path} 不存在，尝试从data.json加载")
        return None

def load_environment_from_json(json_file: str = "data.json") -> Tuple[Dict[int, LEOSatellite], Dict[int, MEOSatellite], dict]:
    """
    从JSON文件加载完整的卫星环境数据（兼容旧版本）

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

def get_leo_meo_assignment(slot_id: int, data: dict = None, meo_per_slot_data: List[Dict] = None) -> List[int]:
    """
    获取指定时间槽的LEO-MEO分配关系

    Args:
        slot_id: 时间槽ID
        data: 原始JSON数据（用于向后兼容）
        meo_per_slot_data: 独立的MEO分配数据

    Returns:
        LEO卫星对应的MEO控制节点ID列表
    """
    # 优先使用独立的MEO分配数据
    if meo_per_slot_data is None:
        meo_per_slot_data = load_meo_per_slot()

    if meo_per_slot_data is not None:
        for slot_info in meo_per_slot_data:
            if slot_info['slot_id'] == slot_id:
                return slot_info['leo_meo_assignments']
    elif data is not None:
        # 向后兼容：从data.json加载
        meo_assignments = data.get('MEO_per_slot', [])
        for slot_info in meo_assignments:
            if slot_info['slot_id'] == slot_id:
                return slot_info['leo_meo_assignments']

    # 如果没找到，返回默认分配
    if data:
        num_leos = data.get('num_satellites', 10)
        num_meos = data.get('num_meo_satellites', 3)
    else:
        print("警告: 无法获取卫星数量配置，使用默认值")
        num_leos = 7
        num_meos = 3

    return [i % num_meos for i in range(num_leos)]

def get_meo_positions_for_slot(slot_id: int, data: dict = None, meo_positions_data: List[List[List[float]]] = None) -> List[List[float]]:
    """
    获取指定时间槽的MEO卫星位置信息

    Args:
        slot_id: 时间槽ID
        data: 原始JSON数据（用于向后兼容）
        meo_positions_data: 独立的MEO位置数据

    Returns:
        MEO卫星位置列表 [[lat, lon, alt], ...]
    """
    # 优先使用独立的MEO位置数据
    if meo_positions_data is None:
        meo_positions_data = load_meo_positions_per_slot()

    if meo_positions_data is not None:
        if slot_id < len(meo_positions_data):
            return meo_positions_data[slot_id]
    elif data is not None:
        # 向后兼容：从data.json加载
        if 'meo_positions_per_slot' in data:
            if slot_id < len(data['meo_positions_per_slot']):
                return data['meo_positions_per_slot'][slot_id]
        # 回退到静态MEO位置
        return data.get('meo_positions', [])

    # 如果都没有，返回空列表
    print(f"警告: 无法获取时间槽 {slot_id} 的MEO位置数据")
    return []

def create_leos_for_slot(slot_id: int, data: dict = None, sat_positions_data: List[List[List[float]]] = None,
                        meo_per_slot_data: List[Dict] = None) -> Dict[int, LEOSatellite]:
    """
    为指定时间槽创建LEO卫星

    Args:
        slot_id: 时间槽ID
        data: 原始JSON数据（用于向后兼容）
        sat_positions_data: 独立的LEO位置数据
        meo_per_slot_data: 独立的MEO分配数据

    Returns:
        LEO卫星字典
    """
    leos = {}

    # 获取LEO位置数据
    if sat_positions_data is None:
        sat_positions_data = load_sat_positions_per_slot()

    sat_positions = None
    if sat_positions_data is not None:
        if slot_id < len(sat_positions_data):
            sat_positions = sat_positions_data[slot_id]
    elif data is not None:
        # 向后兼容：从data.json加载
        if slot_id >= len(data.get('sat_positions_per_slot', [])):
            raise ValueError(f"时间槽 {slot_id} 超出可用数据范围 (0-{len(data['sat_positions_per_slot'])-1})")
        sat_positions = data['sat_positions_per_slot'][slot_id]

    if sat_positions is None:
        raise ValueError(f"无法获取时间槽 {slot_id} 的LEO位置数据")

    # 获取该时间槽的邻居关系
    neighbors_info = None
    if data is not None:
        for neighbor_slot in data.get('neighbors_per_slot', []):
            if neighbor_slot['slot_id'] == slot_id:
                neighbors_info = neighbor_slot['neighbors']
                break

    # 获取MEO分配
    meo_assignments = get_leo_meo_assignment(slot_id, data, meo_per_slot_data)

    # 创建LEO卫星
    for i, (lat, lon) in enumerate(sat_positions):
        neighbors = neighbors_info[i] if neighbors_info and i < len(neighbors_info) else []
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

def create_meos_for_slot(slot_id: int, data: dict = None, meo_positions_data: List[List[List[float]]] = None) -> Dict[int, MEOSatellite]:
    """
    为指定时间槽创建MEO卫星

    Args:
        slot_id: 时间槽ID
        data: 原始JSON数据（用于向后兼容）
        meo_positions_data: 独立的MEO位置数据

    Returns:
        MEO卫星字典
    """
    meos = {}

    # 获取该时间槽的MEO位置
    meo_positions = get_meo_positions_for_slot(slot_id, data, meo_positions_data)

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
    加载指定时间槽的完整环境（支持动态MEO和独立文件）

    Args:
        slot_id: 时间槽ID
        json_file: JSON数据文件路径（用于向后兼容和其他配置）

    Returns:
        (leos, meos, raw_data): 完整的环境数据
    """
    # 尝试加载独立的数据文件
    sat_positions_data = load_sat_positions_per_slot()
    meo_positions_data = load_meo_positions_per_slot()
    meo_per_slot_data = load_meo_per_slot()

    # 加载主配置文件（用于其他配置信息）
    data = None
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)

    # 创建LEO卫星
    leos = create_leos_for_slot(slot_id, data, sat_positions_data, meo_per_slot_data)

    # 创建MEO卫星（动态位置）
    meos = create_meos_for_slot(slot_id, data, meo_positions_data)

    # 更新MEO的cluster信息
    update_meo_clusters(leos, meos)

    return leos, meos, data

def validate_dynamic_meo_data(data: dict = None, meo_positions_data: List[List[List[float]]] = None,
                             sat_positions_data: List[List[List[float]]] = None) -> bool:
    """
    验证动态MEO数据的完整性

    Args:
        data: 原始JSON数据（可选）
        meo_positions_data: 独立的MEO位置数据（可选）
        sat_positions_data: 独立的LEO位置数据（可选）

    Returns:
        数据是否有效
    """
    # 尝试加载独立文件
    if meo_positions_data is None:
        meo_positions_data = load_meo_positions_per_slot()
    if sat_positions_data is None:
        sat_positions_data = load_sat_positions_per_slot()

    # 检查是否有动态MEO位置数据
    if meo_positions_data is None:
        if data is None or 'meo_positions_per_slot' not in data:
            print("警告: 没有找到动态MEO位置数据")
            return False
        meo_positions_data = data['meo_positions_per_slot']

    # 检查LEO位置数据
    if sat_positions_data is None:
        if data is None or 'sat_positions_per_slot' not in data:
            print("警告: 没有找到LEO位置数据")
            return False
        sat_positions_data = data['sat_positions_per_slot']

    # 检查时间槽数量是否匹配
    if len(meo_positions_data) != len(sat_positions_data):
        print(f"警告: MEO位置时间槽数 ({len(meo_positions_data)}) 与LEO位置时间槽数 ({len(sat_positions_data)}) 不匹配")
        return False

    # 检查每个时间槽的MEO数量是否一致
    num_meos = data.get('num_meo_satellites', 3) if data else 3
    for slot_id, meo_positions in enumerate(meo_positions_data):
        if len(meo_positions) != num_meos:
            print(f"警告: 时间槽 {slot_id} 的MEO数量 ({len(meo_positions)}) 与配置不符 ({num_meos})")
            return False

    print(f"动态MEO数据验证成功: {len(meo_positions_data)} 个时间槽，每个时间槽 {num_meos} 个MEO")
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

def create_separate_json_files(data_file: str = "data/data.json"):
    """
    从原始data.json文件创建独立的JSON文件

    Args:
        data_file: 原始数据文件路径
    """
    try:
        with open(data_file, 'r') as f:
            data = json.load(f)

        # 确保data目录存在
        os.makedirs('data', exist_ok=True)

        # 提取并保存sat_positions_per_slot
        if 'sat_positions_per_slot' in data:
            with open('data/sat_positions_per_slot.json', 'w') as f:
                json.dump(data['sat_positions_per_slot'], f, indent=2)
            print("创建文件: data/sat_positions_per_slot.json")

        # 提取并保存meo_positions_per_slot
        if 'meo_positions_per_slot' in data:
            with open('data/meo_positions_per_slot.json', 'w') as f:
                json.dump(data['meo_positions_per_slot'], f, indent=2)
            print("创建文件: data/meo_positions_per_slot.json")

        # 提取并保存MEO_per_slot
        if 'MEO_per_slot' in data:
            with open('data/MEO_per_slot.json', 'w') as f:
                json.dump(data['MEO_per_slot'], f, indent=2)
            print("创建文件: data/MEO_per_slot.json")

        print("独立JSON文件创建完成!")

    except FileNotFoundError:
        print(f"错误: 找不到文件 {data_file}")
    except Exception as e:
        print(f"创建独立文件时出错: {e}")

if __name__ == "__main__":
    # 测试独立文件加载功能

    print("=== 测试独立文件数据加载 ===")

    # 首先检查是否存在独立文件，如果不存在则从data.json创建
    required_files = ['data/sat_positions_per_slot.json', 'data/meo_positions_per_slot.json', 'data/MEO_per_slot.json']
    if not all(os.path.exists(f) for f in required_files):
        print("独立JSON文件不存在，尝试从data/data.json创建...")
        create_separate_json_files('data/data.json')

    # 验证数据
    try:
        is_valid = validate_dynamic_meo_data()

        if not is_valid:
            print("数据验证失败，请检查文件完整性")

        # 测试几个时间槽的数据加载
        print("\n=== 测试独立文件环境加载 ===")
        for slot_id in [0, 2, 4]:
            try:
                leos, meos, data = load_complete_environment(slot_id)
                print_environment_summary(leos, meos, slot_id)
            except Exception as e:
                print(f"加载时间槽 {slot_id} 失败: {e}")

        print("\n=== 测试独立文件加载性能 ===")
        # 测试独立文件加载
        sat_data = load_sat_positions_per_slot()
        meo_data = load_meo_positions_per_slot()
        meo_assign_data = load_meo_per_slot()

        print(f"成功加载独立文件:")
        print(f"  - LEO位置数据: {len(sat_data) if sat_data else 0} 个时间槽")
        print(f"  - MEO位置数据: {len(meo_data) if meo_data else 0} 个时间槽")
        print(f"  - MEO分配数据: {len(meo_assign_data) if meo_assign_data else 0} 个时间槽")

    except Exception as e:
        print(f"测试失败: {e}")