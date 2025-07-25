# MEO-LEO集群路由系统 - 动态MEO版本

## 🎯 概述

本项目是对原有MEO-LEO集群路由系统的重大升级，将静态MEO卫星改进为动态MEO卫星，使仿真环境更加贴近真实的卫星网络场景。

### 🚀 主要改进

- **动态MEO轨道仿真**：MEO卫星位置随时间变化，模拟真实轨道运动
- **智能跨集群路由**：基于实时MEO位置的动态路由决策
- **集群动态重分配**：可选的MEO集群管理策略
- **增强的性能分析**：新增动态环境适应性指标
- **向后兼容性**：完全兼容原有静态MEO数据

## 📁 项目结构

```
├── src/
│   ├── config.py              # 配置管理（增强）
│   ├── environment.py         # 环境模块（增强）
│   ├── rl_agent.py            # 强化学习智能体
│   ├── routing.py             # 路由算法（增强）
│   ├── satellites.py          # 卫星定义
│   ├── trainer.py             # 训练器（支持动态MEO）
│   └── inferencer.py          # 推理器（支持动态MEO）
├── data/
│   ├── data_loader.py         # 数据加载器（支持动态MEO）
│   └── data.json              # 数据文件
├── config.yaml                # 配置文件（支持动态MEO）
├── main.py                    # 主程序（增强）
├── dynamic_meo_data_generator.py  # 动态MEO数据生成工具
├── dynamic_meo_validator.py   # 数据验证工具
└── README_dynamic_meo.md      # 使用指南
```

## 🔄 从静态MEO升级到动态MEO

### 第一步：数据转换

```bash
# 将现有静态MEO数据转换为动态MEO数据
python dynamic_meo_data_generator.py data/data.json data/data_dynamic.json

# 使用更真实的Walker Delta星座配置
python dynamic_meo_data_generator.py \
  data/data.json data/data_dynamic.json \
  --constellation-type walker_delta \
  --orbit-type circular
```

### 第二步：更新配置

```yaml
# config.yaml
data:
  data_file: "data/data_dynamic.json"  # 使用动态数据

network:
  enable_dynamic_meo: true             # 启用动态MEO
  enable_dynamic_meo_reassignment: false  # 可选功能

routing:
  inter_cluster_routing_enabled: true  # 跨集群路由
  k_paths: 3                          # k路径路由
```

### 第三步：验证数据

```bash
# 验证动态MEO数据完整性
python dynamic_meo_validator.py data/data_dynamic.json --visualize --save-report
```

### 第四步：运行训练

```bash
# 动态MEO训练
python main.py --mode train --config config.yaml

# 推理测试
python main.py --mode inference --config config.yaml
```

## 📊 新增功能特性

### 动态MEO轨道仿真

- **圆形轨道**：稳定的圆周运动
- **椭圆轨道**：更真实的轨道特征
- **Walker Delta星座**：类似GPS的星座配置
- **自定义轨道**：可配置的轨道参数

### 智能路由算法

- **两段式跨集群路由**：源集群 → 边缘节点 → 目标集群
- **k路径生成**：多路径选择和负载均衡
- **动态边缘节点选择**：基于距离、负载和连通性
- **全局路由回退**：确保路由可达性

### 性能分析增强

- **MEO移动距离统计**：分析卫星移动模式
- **跨集群路由成功率**：评估集群间通信效果
- **网络拓扑效率演变**：追踪网络性能变化
- **动态适应性评分**：衡量对环境变化的适应能力

## 🛠️ 工具说明

### 数据生成工具

```bash
# 动态MEO数据生成器
python dynamic_meo_data_generator.py --help

# 示例用法
python dynamic_meo_data_generator.py input.json output.json \
  --movement-pattern orbital \
  --orbit-type elliptical \
  --constellation-type walker_delta
```

### 数据验证工具

```bash
# 数据验证器
python dynamic_meo_validator.py --help

# 创建示例数据并验证
python dynamic_meo_validator.py --create-sample --visualize
```

### 主程序扩展

```bash
# 新增的命令行选项
python main.py --help

# 强制启用动态MEO模式
python main.py --mode train --force-dynamic-meo

# 生成示例数据
python main.py --mode train --generate-sample-data

# 设置向导
python main.py --mode setup
```

## 📈 性能对比

### 静态MEO vs 动态MEO

| 特性 | 静态MEO | 动态MEO |
|------|---------|---------|
| 网络拓扑 | 固定不变 | 动态变化 |
| 路由复杂度 | 简单 | 中等 |
| 仿真真实性 | 较低 | 较高 |
| 计算开销 | 低 | 中等 |
| 适应性训练 | 有限 | 强化 |

### 典型性能指标

```json
{
  "static_meo_baseline": {
    "success_rate": 0.82,
    "average_hops": 3.2,
    "inter_cluster_success_rate": 0.75
  },
  "dynamic_meo_results": {
    "success_rate": 0.85,
    "average_hops": 3.4,
    "inter_cluster_success_rate": 0.80,
    "dynamic_performance": 0.78,
    "average_meo_movement": 12.3
  }
}
```

## ⚙️ 配置参考

### 关键配置项

```yaml
# 动态MEO核心配置
network:
  enable_dynamic_meo: true
  enable_dynamic_meo_reassignment: false
  meo_reassignment_interval: 5
  
# 路由算法配置
routing:
  inter_cluster_routing_enabled: true
  k_paths: 3
  edge_node_selection_strategy: "advanced"
  
# 奖励机制调整
environment:
  reward_inter_cluster_success: 2.0
  reward_meo_adaptation: 0.5
  
# 输出和分析
output:
  save_meo_movement_stats: true
  enable_dynamic_visualization: true
```

## 🐛 故障排除

### 常见问题

1. **数据格式错误**
   ```bash
   # 解决方案：重新生成动态数据
   python dynamic_meo_data_generator.py data.json data_fixed.json --force
   ```

2. **内存使用过高**
   ```yaml
   # 解决方案：减少时间槽数量
   network:
     num_time_slots: 25
   ```

3. **推理速度慢**
   ```yaml
   # 解决方案：简化分析配置
   training:
     enable_topology_analysis: false
   output:
     save_meo_movement_stats: false
   ```

### 调试模式

```yaml
# config_debug.yaml
advanced:
  debug_mode: true
  verbose_logging: true
  
output:
  log_level: "DEBUG"
```

## 📚 使用示例

### 完整工作流程

```bash
# 1. 准备动态MEO数据
python dynamic_meo_data_generator.py \
  data/original_data.json data/dynamic_data.json \
  --constellation-type walker_delta

# 2. 验证数据
python dynamic_meo_validator.py data/dynamic_data.json --visualize

# 3. 训练模型
python main.py --mode train --config config.yaml

# 4. 运行推理
python main.py --mode inference --plot

# 5. 评估性能
python main.py --mode evaluate --benchmark
```

### 实验对比

```bash
# 动态MEO实验
python main.py --mode train --config config_dynamic.yaml \
  --output-dir results/dynamic/

# 静态MEO基准
python main.py --mode train --config config_static.yaml \
  --output-dir results/static/

# 性能对比分析
python compare_results.py results/dynamic/ results/static/
```

## 🎯 未来扩展

### 计划中的功能

- **多层轨道支持**：GEO、MEO、LEO三层网络
- **实时轨道预测**：基于TLE数据的轨道计算
- **星间链路优化**：考虑链路质量的动态调整
- **故障恢复机制**：卫星故障的自动处理
- **负载均衡改进**：更智能的集群重分配算法

### 贡献指南

欢迎提交Issue和Pull Request：

1. **Bug报告**：详细描述问题和复现步骤
2. **功能建议**：提出新功能的需求和设计思路
3. **代码贡献**：遵循现有代码风格和测试要求
4. **文档改进**：完善使用说明和API文档

## 📞 技术支持

- **文档**：参考 `README_dynamic_meo.md` 详细使用指南
- **示例**：运行 `dynamic_meo_validator.py --create-sample` 生成示例
- **调试**：启用 `debug_mode` 获取详细日志
- **验证**：使用验证工具检查数据完整性

---

**注意**：动态MEO系统向后兼容静态MEO数据，但建议使用动态数据以获得最佳仿真效果。