"""
基于分离的JSON文件的数据加载和环境初始化模块
包含动态MEO卫星信息和LEO-MEO分配关系
支持从独立的JSON文件加载：sat_positions_per_slot.json, meo_positions_per_slot.json, MEO_per_slot.json
支持从独立的neighbors文件夹加载邻居关系数据
"""

import json
import os
from typing import Dict, Tuple, List, Optional
import sys

# 获取当前文件所在目录（data目录）
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 动态添加src目录到路径，以便导入satellites模块
SRC_DIR = os.path.join(os.path.dirname(CURRENT_DIR), 'src')
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

try:
    from satellites import LEOSatellite, MEOSatellite
except ImportError:
    # 如果直接导入失败，尝试其他路径
    sys.path.append(os.path.dirname(CURRENT_DIR))
    from src.satellites import LEOSatellite, MEOSatellite

def get_data_file_path(filename: str) -> str:
    """获取data目录下文件的完整路径"""
    return os.path.join(CURRENT_DIR, filename)


def create_meo_id_mapping(meo_assignments: List[int], num_meos: int = 32) -> Tuple[Dict[int, int], List[int]]:
    """创建MEO ID映射，将大的MEO ID映射到0-31的本地索引"""
    unique_meo_ids = list(set(meo_assignments))
    unique_meo_ids.sort()

    meo_mapping = {}
    for i, meo_id in enumerate(unique_meo_ids):
        meo_mapping[meo_id] = i % num_meos

    mapped_assignments = [meo_mapping[meo_id] for meo_id in meo_assignments]

    print(f"MEO ID映射: 发现 {len(unique_meo_ids)} 个唯一MEO ID，映射到 {num_meos} 个本地索引")
    return meo_mapping, mapped_assignments


def get_leo_meo_assignment_with_mapping(slot_id: int, meo_per_slot_data: List[Dict] = None) -> List[int]:
    """获取LEO-MEO分配关系（带ID映射）"""
    raw_assignments = get_leo_meo_assignment(slot_id, meo_per_slot_data)
    meo_mapping, mapped_assignments = create_meo_id_mapping(raw_assignments)
    return raw_assignments

def load_sat_positions_per_slot(file_path: str = None) -> Optional[List[List[List[float]]]]:
    """
    从独立文件加载LEO卫星位置数据

    Args:
        file_path: sat_positions_per_slot.json文件路径

    Returns:
        LEO卫星位置数据 [slot][satellite_id][lat, lon]
    """
    if file_path is None:
        file_path = get_data_file_path("sat_positions_per_slot.json")

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"成功加载LEO位置数据: {len(data)} 个时间槽")
        return data
    except FileNotFoundError:
        print(f"警告: 文件 {file_path} 不存在")
        return None
    except json.JSONDecodeError as e:
        print(f"错误: LEO位置数据文件格式错误 - {e}")
        return None
    except Exception as e:
        print(f"错误: 加载LEO位置数据失败 - {e}")
        return None

def load_meo_positions_per_slot(file_path: str = None) -> Optional[List[List[List[float]]]]:
    """
    从独立文件加载MEO卫星位置数据

    Args:
        file_path: meo_positions_per_slot.json文件路径

    Returns:
        MEO卫星位置数据 [slot][meo_id][lat, lon, alt]
    """
    if file_path is None:
        file_path = get_data_file_path("meo_positions_per_slot.json")

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"成功加载MEO位置数据: {len(data)} 个时间槽")
        return data
    except FileNotFoundError:
        print(f"警告: 文件 {file_path} 不存在")
        return None
    except json.JSONDecodeError as e:
        print(f"错误: MEO位置数据文件格式错误 - {e}")
        return None
    except Exception as e:
        print(f"错误: 加载MEO位置数据失败 - {e}")
        return None

def load_meo_per_slot(file_path: str = None) -> Optional[List[Dict]]:
    """
    从独立文件加载LEO-MEO分配关系数据

    Args:
        file_path: MEO_per_slot.json文件路径

    Returns:
        LEO-MEO分配数据 [{"slot_id": int, "leo_meo_assignments": [int]}]
    """
    if file_path is None:
        file_path = get_data_file_path("MEO_per_slot.json")

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"成功加载LEO-MEO分配数据: {len(data)} 个时间槽")
        return data
    except FileNotFoundError:
        print(f"警告: 文件 {file_path} 不存在")
        return None
    except json.JSONDecodeError as e:
        print(f"错误: LEO-MEO分配数据文件格式错误 - {e}")
        return None
    except Exception as e:
        print(f"错误: 加载LEO-MEO分配数据失败 - {e}")
        return None

def load_neighbors_for_slot(slot_id: int, neighbors_dir: str = None) -> List[List[int]]:
    """
    从独立文件加载指定时间槽的邻居关系数据

    Args:
        slot_id: 时间槽ID
        neighbors_dir: neighbors文件夹路径

    Returns:
        邻居关系列表
    """
    if neighbors_dir is None:
        neighbors_dir = get_data_file_path("neighbors")

    neighbor_file = os.path.join(neighbors_dir, f"slot_{slot_id}.json")

    try:
        with open(neighbor_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"找不到时间槽 {slot_id} 的neighbor文件: {neighbor_file}")
    except json.JSONDecodeError as e:
        raise ValueError(f"neighbor文件格式错误 {neighbor_file}: {e}")
    except Exception as e:
        raise RuntimeError(f"加载neighbor文件失败 {neighbor_file}: {e}")

def load_main_config(json_file: str = None) -> dict:
    """
    加载主配置文件（data.json）

    Args:
        json_file: 主配置文件路径

    Returns:
        配置数据字典
    """
    if json_file is None:
        json_file = get_data_file_path("data.json")

    try:
        with open(json_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"主配置文件不存在: {json_file}")
    except json.JSONDecodeError as e:
        raise ValueError(f"主配置文件格式错误: {e}")
    except Exception as e:
        raise RuntimeError(f"加载主配置文件失败: {e}")

def get_leo_meo_assignment(slot_id: int, meo_per_slot_data: List[Dict] = None) -> List[int]:
    """
    获取指定时间槽的LEO-MEO分配关系

    Args:
        slot_id: 时间槽ID
        meo_per_slot_data: 独立的MEO分配数据

    Returns:
        LEO卫星对应的MEO控制节点ID列表
    """
    # 优先使用传入的数据，否则加载独立文件
    if meo_per_slot_data is None:
        meo_per_slot_data = load_meo_per_slot()

    if meo_per_slot_data is not None:
        for slot_info in meo_per_slot_data:
            if slot_info['slot_id'] == slot_id:
                return slot_info['leo_meo_assignments']

    else:
        raise Exception("读取的MEO_per_slot.json文件为空数据")

def get_meo_positions_for_slot(slot_id: int, meo_positions_data: List[List[List[float]]] = None) -> List[List[float]]:
    """
    获取指定时间槽的MEO卫星位置信息

    Args:
        slot_id: 时间槽ID
        meo_positions_data: 独立的MEO位置数据

    Returns:
        MEO卫星位置列表 [[lat, lon, alt], ...]
    """
    # 优先使用传入的数据，否则加载独立文件
    if meo_positions_data is None:
        meo_positions_data = load_meo_positions_per_slot()

    if meo_positions_data is not None:
        if slot_id < len(meo_positions_data):
            return meo_positions_data[slot_id]
        else:
            raise IndexError(f"时间槽 {slot_id} 超出MEO位置数据范围 (0-{len(meo_positions_data)-1})")

    else:
        raise Exception("读取meo_positions_per_slot.json文件返回为空")

def get_sat_positions_for_slot(slot_id: int, sat_positions_data: List[List[List[float]]] = None) -> List[List[float]]:
    """
    获取指定时间槽的LEO卫星位置信息

    Args:
        slot_id: 时间槽ID
        sat_positions_data: 独立的LEO位置数据

    Returns:
        LEO卫星位置列表 [[lat, lon], ...]
    """
    # 优先使用传入的数据，否则加载独立文件
    if sat_positions_data is None:
        sat_positions_data = load_sat_positions_per_slot()

    if sat_positions_data is not None:
        if slot_id < len(sat_positions_data):
            return sat_positions_data[slot_id]
        else:
            raise IndexError(f"时间槽 {slot_id} 超出LEO位置数据范围 (0-{len(sat_positions_data)-1})")
    else:
        raise Exception("从sat_positions_per_slot.json加载的数据是空的")


def create_leos_for_slot(slot_id: int,
                        sat_positions_data: List[List[List[float]]] = None,
                        meo_per_slot_data: List[Dict] = None,
                        neighbors_dir: str = None) -> Dict[int, LEOSatellite]:
    """
    为指定时间槽创建LEO卫星（修复版本，处理位置数据维度问题）

    Args:
        slot_id: 时间槽ID
        sat_positions_data: 独立的LEO位置数据
        meo_per_slot_data: 独立的MEO分配数据
        neighbors_dir: neighbors文件夹路径

    Returns:
        LEO卫星字典
    """
    leos = {}

    # 获取LEO位置数据
    sat_positions = get_sat_positions_for_slot(slot_id, sat_positions_data)

    # 加载邻居关系
    if neighbors_dir is None:
        neighbors_dir = get_data_file_path("neighbors")
    neighbors_info = load_neighbors_for_slot(slot_id, neighbors_dir)

    # 获取MEO分配
    meo_assignments = get_leo_meo_assignment_with_mapping(slot_id, meo_per_slot_data)

    # 创建LEO卫星 - 修复：处理2D和3D位置数据
    for i, position in enumerate(sat_positions):
        neighbors = neighbors_info[i] if i < len(neighbors_info) else []
        meo_id = meo_assignments[i] if i < len(meo_assignments) else 0

        # 处理不同维度的位置数据
        if isinstance(position, (list, tuple)):
            if len(position) >= 3:
                # 3D位置数据 [lat, lon, alt]
                lat, lon, alt = float(position[0]), float(position[1]), float(position[2])
            elif len(position) == 2:
                # 2D位置数据 [lat, lon]
                lat, lon = float(position[0]), float(position[1])
                alt = 500.0  # 默认LEO高度
            else:
                raise ValueError(f"LEO位置数据格式错误，卫星 {i}: {position}")
        else:
            raise ValueError(f"LEO位置数据类型错误，卫星 {i}: {type(position)} - {position}")

        leos[i] = LEOSatellite(
            id=i,
            latitude=lat,
            longitude=lon,
            altitude=alt,
            load=0,  # 初始负载为0
            neighbors=neighbors,
            meo_id=meo_id
        )

    print(f"时间槽 {slot_id}: 创建了 {len(leos)} 个LEO卫星")
    return leos

def create_meos_for_slot(slot_id: int, meo_positions_data: List[List[List[float]]] = None) -> Dict[int, MEOSatellite]:
    """
    为指定时间槽创建MEO卫星

    Args:
        slot_id: 时间槽ID
        meo_positions_data: 独立的MEO位置数据

    Returns:
        MEO卫星字典
    """
    meos = {}

    # 获取该时间槽的MEO位置
    meo_positions = get_meo_positions_for_slot(slot_id, meo_positions_data)

    # 创建MEO卫星
    for i, position in enumerate(meo_positions):
        if len(position) >= 3:
            lat, lon, alt = position[0], position[1], position[2]
        else:
            # 兼容只有2D位置的情况
            lat, lon = position[0], position[1]
            alt = 1000.0  # 默认MEO高度

        meos[i] = MEOSatellite(
            id=i + 1723, # 为了迎合跟leo统一排序的id
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

def load_complete_environment(slot_id: int,
                             json_file: str = None,
                             neighbors_dir: str = None) -> Tuple[Dict[int, LEOSatellite], Dict[int, MEOSatellite], dict]:
    """
    加载指定时间槽的完整环境（优先使用独立文件）

    Args:
        slot_id: 时间槽ID
        json_file: JSON数据文件路径（主要用于配置信息）
        neighbors_dir: neighbors文件夹路径

    Returns:
        (leos, meos, config_data): 完整的环境数据
    """
    # 预加载所有独立数据文件
    sat_positions_data = load_sat_positions_per_slot()
    meo_positions_data = load_meo_positions_per_slot()
    meo_per_slot_data = load_meo_per_slot()

    # 加载主配置文件（用于其他配置信息）
    try:
        config_data = load_main_config(json_file)
    except Exception as e:
        print(f"警告: 无法加载主配置文件 - {e}")
        config_data = {}

    # 设置默认neighbors目录
    if neighbors_dir is None:
        raise Exception("data目录下没有neighbors")

    # 创建LEO卫星
    leos = create_leos_for_slot(slot_id, sat_positions_data, meo_per_slot_data, neighbors_dir)

    # 创建MEO卫星
    meos = create_meos_for_slot(slot_id, meo_positions_data)

    # 更新MEO的cluster信息
    update_meo_clusters(leos, meos)

    return leos, meos, config_data


def validate_dynamic_meo_data(data: dict = None) -> bool:
    """
    验证动态MEO数据的完整性（修复版本 - 正确处理参数）

    Args:
        data: 主配置数据字典（从data.json加载的内容）

    Returns:
        数据是否有效
    """
    print("验证动态MEO数据完整性...")
    # 直接加载独立文件，不依赖传入参数
    meo_positions_data = load_meo_positions_per_slot()
    sat_positions_data = load_sat_positions_per_slot()

    # 检查是否有动态MEO位置数据
    if meo_positions_data is None:
        raise Exception("警告: 没有找到独立的MEO位置数据文件")

    # 检查LEO位置数据
    if sat_positions_data is None:
        raise Exception("警告: 没有找到独立的MEO位置数据文件")

    # 检查时间槽数量匹配
    meo_slots = len(meo_positions_data)
    leo_slots = len(sat_positions_data)

    print(f"MEO位置数据时间槽数: {meo_slots}")
    print(f"LEO位置数据时间槽数: {leo_slots}")

    if meo_slots != leo_slots:
        raise Exception("meo和leo位置文件的时间槽数量不一样")
    else:
        min_slots = meo_slots

    # 获取配置信息验证数量
    try:
        if data is None:
            data = load_main_config()
        expected_meos = data.get('num_meo_satellites')
        expected_leos = data.get('num_satellites')
    except Exception:
        raise Exception("data.json文件中没有num_meo_satellites和num_satellites元素")

    # 检查前几个时间槽的数据完整性
    check_slots = min(3, min_slots)  # 只检查前3个时间槽，减少输出
    for slot_id in range(check_slots):
        # 检查MEO数量
        if slot_id < len(meo_positions_data):
            meo_count = len(meo_positions_data[slot_id])
            if meo_count != expected_meos:
                print(f"警告: 时间槽 {slot_id} 的MEO数量 ({meo_count}) 与配置不符 ({expected_meos})")

        # 检查LEO数量
        if slot_id < len(sat_positions_data):
            leo_count = len(sat_positions_data[slot_id])
            if leo_count != expected_leos:
                print(f"警告: 时间槽 {slot_id} 的LEO数量 ({leo_count}) 与配置不符 ({expected_leos})")

    print(f"✅ 动态MEO数据验证完成: 可用时间槽 {min_slots} 个")
    return True

def validate_neighbors_data(neighbors_dir: str = None, expected_slots: int = None) -> bool:
    """
    验证neighbors文件夹中的数据完整性

    Args:
        neighbors_dir: neighbors文件夹路径
        expected_slots: 期望的时间槽数量

    Returns:
        数据是否有效
    """
    if neighbors_dir is None:
        neighbors_dir = get_data_file_path("neighbors")

    if not os.path.exists(neighbors_dir):
        print(f"错误: neighbors目录 {neighbors_dir} 不存在")
        return False

    # 获取所有slot文件
    slot_files = [f for f in os.listdir(neighbors_dir) if f.startswith('slot_') and f.endswith('.json')]

    if not slot_files:
        print(f"错误: 在 {neighbors_dir} 中没有找到slot文件")
        return False

    slot_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))  # 按slot_id排序

    print(f"找到 {len(slot_files)} 个neighbors文件")

    if expected_slots and len(slot_files) != expected_slots:
        print(f"警告: 期望 {expected_slots} 个文件，实际找到 {len(slot_files)} 个")

    # 验证每个文件
    valid_files = 0
    for slot_file in slot_files:
        slot_path = os.path.join(neighbors_dir, slot_file)
        try:
            with open(slot_path, 'r') as f:
                neighbors = json.load(f)

            if isinstance(neighbors, list):
                valid_files += 1
            else:
                print(f"警告: {slot_file} 格式错误，应为列表")

        except Exception as e:
            print(f"错误: 读取 {slot_file} 失败 - {e}")

    print(f"有效的neighbors文件: {valid_files}/{len(slot_files)}")
    return valid_files == len(slot_files)

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

if __name__ == "__main__":
    # 测试独立文件加载功能
    print("=== 测试修改后的数据加载功能 ===")

    # 验证neighbors文件夹
    print("\n1. 验证neighbors文件夹:")
    neighbors_valid = validate_neighbors_data()

    if not neighbors_valid:
        print("错误: neighbors数据不完整")
    else:
        print("✅ neighbors数据验证通过")

    # 验证动态MEO数据
    print("\n2. 验证动态MEO数据:")
    is_valid = validate_dynamic_meo_data()

    if not is_valid:
        print("警告: 动态MEO数据验证失败，请检查文件完整性")
    else:
        print("✅ 动态MEO数据验证通过")

        # 测试几个时间槽的数据加载
        print("\n3. 测试环境加载:")
        for slot_id in [0, 1, 2]:
            try:
                print(f"\n--- 加载时间槽 {slot_id} ---")
                leos, meos, config_data = load_complete_environment(slot_id)
                print_environment_summary(leos, meos, slot_id)
            except Exception as e:
                print(f"加载时间槽 {slot_id} 失败: {e}")

    print(f"\n=== 测试完成 ===")
    print("data_loader.py 已修改为优先使用独立的JSON文件")