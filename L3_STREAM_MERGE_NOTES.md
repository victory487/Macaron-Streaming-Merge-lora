# Macaron-V1-Venti L3 流式合并方案笔记

## 1. 目标

将 Macaron-V1-Venti 的 **L3 LoRA** 固化进 BF16 基础权重，生成可直接加载的
标准 Hugging Face Safetensors 模型。

固定路径：

```text
基础模型  /cpfs01/models/Macaron-V1-Venti
L3 LoRA   /cpfs01/models/Macaron-V1-Venti/loras/L3
输出模型  /cpfs01/models/Macaron-V1-Venti-merged-L3
方案目录  /cpfs01/hy/macaron-stream-merge
```

本方案只读取 `loras/L3`，不会合并 L0、L1、L2 或其他 LoRA。

## 2. 为什么不用 LlamaFactory 的常规导出

基础模型共有 282 个 BF16 分片，大小约 1.37 TiB。LlamaFactory 常规
`export` 会通过 Transformers 实例化完整模型，并在加载、设备分配和 LoRA
合并期间保留大量 CPU 暂存权重。此前进程在加载约 64%～66% 时，CPU 匿名
内存接近 895 GiB，最终被系统 `Killed`。

限制 GPU/CPU 的 `max_memory` 主要控制模型权重最终放置位置，不能完全消除
Transformers 加载期间的 CPU 峰值，所以改用不实例化完整模型的分片流式方案。

## 3. 流式合并原理

核心程序：

```text
/cpfs01/hy/macaron-stream-merge/stream_merge_lora_safetensors.py
```

处理流程：

1. 读取 `model.safetensors.index.json`，建立“权重名称 → 分片文件”的映射。
2. 扫描 L3 的 `adapter_model.safetensors`，配对全部 LoRA A/B 矩阵。
3. 在正式写入前校验基础分片、权重名称、rank、dtype 和矩阵形状。
4. 每次只打开一个约 5 GiB 的基础分片。
5. 对该分片涉及的线性层计算 LoRA delta 并合并。
6. 先写入隐藏临时分片，校验键、形状和 dtype 后再原子重命名。
7. 在 `merge_info.json` 中记录已完成分片，支持中断后续跑。
8. 复制模型配置、Tokenizer、模板和索引等元数据，最终保持 282 个原始分片名。

LoRA 合并公式：

```text
W_merged = W_base + scale × (B @ A)
scale = lora_alpha / rank = 32 / 16 = 2.0
```

为匹配 PEFT 在 CPU 上的合并顺序，程序使用 FP32 计算 `B @ A`，将 delta
转换为 BF16 后，再与 BF16 基础权重相加。大矩阵按行分块计算，默认
`row_chunk=2048`，避免产生完整 FP32 delta。

## 4. 已校验的数据

```text
基础模型分片             282
基础模型大小             约 1.37 TiB
L3 LoRA A/B 矩阵对       58,224
被 L3 修改的分片         279
不需要修改的分片         3
基础权重 dtype           BF16
LoRA dtype               BF16
LoRA rank                16
LoRA alpha               32
LoRA scale               2.0
```

全部 58,224 对 LoRA 矩阵均能映射到基础模型，没有缺失键、不完整 A/B 对或
不支持的张量形状。小型端到端测试确认合并结果与 PEFT CPU 运算顺序逐元素一致。

## 5. 资源分配

启动脚本明确设置：

```bash
CUDA_VISIBLE_DEVICES=""
OMP_NUM_THREADS=64
MKL_NUM_THREADS=64
MALLOC_ARENA_MAX=4
```

当前版本是 **CPU 流式合并**，不会使用 4 张 GB300。GPU 利用率为 0% 是正常
现象，并非程序没有运行。仅删除 `CUDA_VISIBLE_DEVICES` 也不会自动使用 GPU，
因为 Safetensors 张量和矩阵计算在程序中明确放在 CPU。

内存只需容纳当前基础分片、当前合并结果和少量 LoRA 矩阵，远低于完整模型加载
所需内存。Linux 的 `buff/cache` 会因连续读取 1.37 TiB 权重而增大，它属于可
回收文件缓存，不等于进程实际常驻内存。

实测单个约 5 GiB 分片耗时约 12.9 秒，输出速度约 390 MiB/s；算上读取基础
分片，总文件读写吞吐约 780 MiB/s。完整合并预计约 1 小时，主要瓶颈是 CPFS
读取和写入，而不是显存。

## 6. 使用命令

### 只读预检

预检不会创建输出目录：

```bash
cd /cpfs01/hy/macaron-stream-merge
./merge_l3.sh --check
```

### 正式合并

```bash
cd /cpfs01/hy/macaron-stream-merge
./merge_l3.sh
```

启动脚本固定使用 LlamaFactory 已安装依赖的 Python：

```text
/cpfs01/hy/LlamaFactory/.venv/bin/python
```

这里仅复用 Python 环境，并不进入 LlamaFactory/Transformers 的全模型加载流程。

## 7. 查看日志和进度

查找并持续查看最新日志：

```bash
cd /cpfs01/hy/macaron-stream-merge
latest_log=$(ls -1t logs/stream_merge_l3_*.log | head -n 1)
tail -n 100 -f "$latest_log"
```

日志中的：

```text
[57/282] merge model-00057-of-00282.safetensors: 205 tensors
```

表示第 57 个分片已经开始处理，不表示已经完成落盘。确认已完成数量应查看断点
标记：

```bash
/cpfs01/hy/LlamaFactory/.venv/bin/python -c '
import json
p = "/cpfs01/models/Macaron-V1-Venti-merged-L3/merge_info.json"
d = json.load(open(p))
print("status:", d["status"])
print("completed:", len(d["completed_shards"]), "/ 282")
'
```

也可以查看已经原子落盘的分片数：

```bash
find /cpfs01/models/Macaron-V1-Venti-merged-L3 \
  -maxdepth 1 -name 'model-*-of-*.safetensors' | wc -l
```

## 8. 中断与续跑

每个完成的分片先经过校验，再从临时文件原子替换为正式文件；随后更新
`merge_info.json`。如果 SSH 中断、进程被终止或节点重启，重新执行相同命令即可：

```bash
cd /cpfs01/hy/macaron-stream-merge
./merge_l3.sh
```

程序会校验并跳过已完成分片。残留的隐藏 `.tmp` 分片不会被当作完成结果，处理
到对应分片时会重新生成。

不要手工修改已经落盘的输出分片或 `merge_info.json`。输出目录存在但没有合法
断点标记时，程序会拒绝复用，避免覆盖来源不明的数据。

## 9. 完成判定与验收

日志末尾应出现：

```text
Merge completed: /cpfs01/models/Macaron-V1-Venti-merged-L3
```

断点标记应显示：

```text
status: complete
completed: 282 / 282
```

检查输出分片数量：

```bash
find /cpfs01/models/Macaron-V1-Venti-merged-L3 \
  -maxdepth 1 -name 'model-*-of-*.safetensors' | wc -l
```

预期输出为 `282`。随后可以确认索引及关键配置存在：

```bash
test -f /cpfs01/models/Macaron-V1-Venti-merged-L3/model.safetensors.index.json
test -f /cpfs01/models/Macaron-V1-Venti-merged-L3/config.json
test -f /cpfs01/models/Macaron-V1-Venti-merged-L3/chat_template.jinja
```

输出模型继承基础模型自身的 `chat_template.jinja`，无需在合并阶段额外指定
LlamaFactory `template`。推理部署时仍应按该模型架构及上下文长度配置推理引擎。

## 10. 文件清单

```text
macaron-stream-merge/
├── stream_merge_lora_safetensors.py  # 核心分片流式合并器
├── merge_l3.sh                       # 固定路径启动脚本
├── requirements.txt                  # 最小依赖说明
├── README.md                         # 简要使用说明
├── L3_STREAM_MERGE_NOTES.md          # 本笔记
└── logs/                             # 正式运行日志
```

