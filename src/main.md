# 修改后main.py的运行命令大全

## 基本语法
```bash
python main.py [参数选项]
```

## 可用参数

### 必选/核心参数
- `--mode`: 运行模式（必选其一）
  - `train`: 训练模式
  - `inference`: 推理模式  
  - `evaluate`: 评估模式
  - `data`: 数据加载演示模式

### 可选参数
- `--config`: 配置文件路径（默认: `config.yaml`）
- `--model`: 模型文件路径（inference和evaluate模式需要）
- `--use-predict-data`: 推理时使用预测数据集（默认启用）
- `--use-train-data`: 推理时使用训练数据集
- `--output-dir`: 结果输出目录
- `--plot`: 生成结果图表（默认启用）

## 具体运行命令示例

### 1. 训练模式
```bash
# 基本训练（使用默认配置）
python main.py --mode train

# 使用自定义配置文件训练
python main.py --mode train --config my_config.yaml

# 训练并指定输出目录
python main.py --mode train --output-dir results/exp1

# 训练但不生成图表
python main.py --mode train --plot=False
```

### 2. 推理模式
```bash
# 基本推理（使用默认模型和预测数据）
python main.py --mode inference

# 使用指定模型进行推理
python main.py --mode inference --model results/final_model.json

# 使用训练数据进行推理
python main.py --mode inference --use-train-data

# 推理并指定输出目录
python main.py --mode inference --model my_model.json --output-dir results/inference

# 推理但不生成图表
python main.py --mode inference --model my_model.json --plot=False
```

### 3. 评估模式
```bash
# 基本评估（在训练集和预测集上评估）
python main.py --mode evaluate

# 使用指定模型评估
python main.py --mode evaluate --model results/final_model.json

# 评估并指定输出目录
python main.py --mode evaluate --model my_model.json --output-dir results/evaluation

# 评估但不生成图表
python main.py --mode evaluate --model my_model.json --plot=False
```

### 4. 数据演示模式
```bash
# 基本数据加载演示
python main.py --mode data

# 使用自定义配置的数据演示
python main.py --mode data --config my_config.yaml
```

## 完整命令组合示例

### 完整训练流程
```bash
# 1. 验证数据并训练
python main.py --mode train --config config.yaml --output-dir results/experiment1

# 2. 使用训练好的模型进行推理
python main.py --mode inference --model results/experiment1/final_model.json --output-dir results/experiment1/inference

# 3. 评估模型性能
python main.py --mode evaluate --model results/experiment1/final_model.json --output-dir results/experiment1/evaluation
```

### 快速验证流程
```bash
# 1. 先检查数据是否正确
python main.py --mode data

# 2. 如果数据正确，开始训练
python main.py --mode train
```

### 模型测试流程
```bash
# 在预测数据上测试
python main.py --mode inference --model results/final_model.json --use-predict-data

# 在训练数据上测试（检查过拟合）
python main.py --mode inference --model results/final_model.json --use-train-data

# 全面评估
python main.py --mode evaluate --model results/final_model.json
```

## 重要注意事项

### 必需的数据文件
运行任何模式前，确保`data/`目录包含：
- `sat_positions_per_slot.json`
- `meo_positions_per_slot.json` 
- `MEO_per_slot.json`
- `data.json`

### 模式依赖关系
- `inference`和`evaluate`模式需要先有训练好的模型
- 如果不指定`--model`，系统会寻找`results/final_model.json`
- `train`模式会生成模型文件供后续使用

### 常见错误处理
```bash
# 如果缺少数据文件
python main.py --mode data  # 先检查数据状态

# 如果找不到模型文件
python main.py --mode train  # 先训练生成模型

# 如果配置文件问题
python main.py --config path/to/valid/config.yaml
```

## 获取帮助
```bash
# 查看所有可用参数
python main.py --help

# 查看特定模式的详细说明
python main.py --mode train --help
```