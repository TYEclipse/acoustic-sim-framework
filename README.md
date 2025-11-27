# Acoustic Simulation Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](#)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](#)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](#)

> 高度可配置的 Python 声学仿真框架，用于生成用于车内噪声定位与分离模型的合成训练数据（8 通道，48kHz，24-bit）。

---

## TL;DR

- 自动生成大量带像素级真值（位置、方向、干净信号）的合成音频样本  
- 可配置车舱声学、麦克风阵列、多个并发声源与频域/时域特性  
- 输出：8 通道 WAV、单声源干净信号、JSON/YAML 标签

---

## 目录

- [主要特性](#主要特性)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [仓库结构](#仓库结构)
- [示例用法](#示例用法)
- [输出格式](#输出格式)
- [测试与质量保证](#测试与质量保证)
- [贡献](#贡献)
- [许可](#许可)

---

## 主要特性

- 多达 8+ 类型的噪声源（发动机、道路、风噪、HVAC、马达嗡嗡、BSR、语音、提示音）
- 并发多源（支持 3+ 并发）与频谱/时域可控
- 车舱几何与材料频率相关吸声特性
- 基于 Image Source Method 的 RIR（可调反射阶数）
- 输出：8 通道 WAV（48kHz/24-bit）+ 完整标签（位置、方位、干净信号路径等）

---

## 快速开始

1. 克隆代码并安装依赖：

```bash
git clone https://github.com/yourusername/acoustic-sim-framework.git
cd acoustic-sim-framework
pip install -r requirements.txt
# 或以开发模式安装
pip install -e .
```

2. 运行单次仿真示例（验证安装）：

```bash
python examples/run_simulation_example.py
```

输出示例：

- `output/example/example_001_mixed.wav` (8ch mixed)
- `output/example/example_001_source_*.wav` (clean sources)
- `output/example/example_001_label.json`

3. 批量生成数据集：

```bash
python scripts/generate_dataset.py \
  --config config/default.yaml \
  --output output/dataset \
  --num-samples 100 \
  --num-workers 4 \
  --seed 42
```

---

## 配置说明（示例片段）

关键配置位于 `config/default.yaml`，示例：

```yaml
audio:
  sampling_rate: 48000
  bit_depth: 24
  duration: 5.0

microphone_array:
  name: "default_8mic_array"
  positions:
    - [0.10, 0.05, 0.0]
    - [0.10, -0.05, 0.0]
    # ... 6 more

room:
  dimensions: [4.5, 1.8, 1.5]
  absorption:
    default: [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
  max_order: 15
  rt60_range: [0.10, 0.20]
```

---

## 仓库结构（简览）

```
acoustic_sim_framework/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   └── default.yaml
├── src/
│   ├── signal_generation/
│   │   └── noise_sources.py
│   ├── propagation_model/
│   │   └── room_impulse_response.py
│   ├── array_geometry/
│   │   └── microphone_array.py
│   ├── data_synthesis/
│   │   └── mixer.py
│   └── utils/
│       ├── audio_io.py
│       └── labels.py
├── examples/
│   └── run_simulation_example.py
├── tests/
└── scripts/
    └── generate_dataset.py
```

---

## 示例用法

示例 1：自定义麦克风阵列

```python
from src.array_geometry.microphone_array import MicrophoneArray

positions = [
    [0.0, 0.1, 0.0],
    [0.0, -0.1, 0.0],
    [0.5, 0.5, 0.2],
    [0.5, -0.5, 0.2]
]

mic_array = MicrophoneArray(positions, name="custom_4mic")
print(mic_array.visualize_array())
```

示例 2：生成特定噪声

```python
from src.signal_generation.noise_sources import NoiseSourceGenerator
generator = NoiseSourceGenerator(sampling_rate=48000)

engine_signal = generator.generate_source(
    source_type='engine_noise',
    duration=5.0,
    frequency_range=(100, 3000),
    volume=0.8,
    rpm=2000
)
```

示例 3：仿真声学场景

```python
from src.propagation_model.room_impulse_response import RoomAcousticSimulator
import numpy as np

# ... 初始化 mic_array 和 engine_signal ...

room_sim = RoomAcousticSimulator(
    room_dimensions=[4.0, 1.8, 1.4],
    mic_array=mic_array,
    sampling_rate=48000,
    max_order=12
)

room_sim.add_source(position=np.array([1.5, 0.0, -0.5]), signal=engine_signal)
mixed_audio = room_sim.simulate()
```

---

## 输出格式（标签 JSON 示例）

```json
{
  "clip_id": "sim_000001",
  "audio_filepath": "/path/to/sim_000001_mixed.wav",
  "sampling_rate": 48000,
  "bit_depth": 24,
  "mic_array_setup": {
    "name": "default_8mic_array",
    "positions": [[0.1, 0.05, 0.0], ...]
  },
  "room_properties": {
    "dimensions": [4.5, 1.8, 1.5],
    "rt60_s": 0.15
  },
  "sources": [
    {
      "source_id": 0,
      "label": "engine_noise",
      "position_xyz": [1.5, 0.2, -0.5],
      "orientation_az_el": [15.0, -10.0],
      "distance_m": 1.8,
      "clean_signal_path": "/path/to/source_0_engine_noise.wav"
    }
  ]
}
```

---

## 测试与质量保证

- 单元测试：`python -m pytest tests/`
- 验证策略：
  1. 物理验证：与真实录音比较光谱与空间特征
  2. 模型验证：在合成数据上训练基线模型并在真实数据上测试
  3. 混合训练：合成 + 真实数据混合以提高泛化

---

## 贡献

1. Fork 仓库
2. 新建分支：`git checkout -b feature/your-feature`
3. 提交并推 PR

---

## 许可

MIT License — 详情请查看 LICENSE 文件。

---

如果需要，我可以：

- 将部分内容翻译为中文或双语版本；
- 添加示例输出图片（提供路径或样例图片即可）；
- 根据演示场景裁剪 README（如简洁版或教学版）。
