# Macaron-V1-Venti L3 流式合并


此目录提供逐 Safetensors 分片的 LoRA 合并器，用于把
`Macaron-V1-Venti/loras/L3` 固化进 BF16 基础权重。它不会构造完整的
Transformers 模型，也不会同时把 1.37 TiB 权重加载进 CPU 内存。

## 固定路径

- 基础模型：`/cpfs01/models/Macaron-V1-Venti`
- L3 adapter：`/cpfs01/models/Macaron-V1-Venti/loras/L3`
- 输出模型：`/cpfs01/models/Macaron-V1-Venti-merged-L3`
- Python：`/cpfs01/hy/LlamaFactory/.venv/bin/python`

## 使用

先执行只读预检：

```bash
cd /cpfs01/hy/macaron-stream-merge
./merge_l3.sh --check
```

正式合并：

```bash
./merge_l3.sh
```

查看最新日志：

```bash
latest_log=$(ls -1t logs/stream_merge_l3_*.log | head -n 1)
tail -n 100 -f "$latest_log"
```

## 行为

- 默认使用 CPU，明确隐藏 GPU，避免完整模型加载路径再次发生 OOM。
- 先验证全部 adapter 键、A/B 配对、rank、基础权重映射和张量形状。
- 合并顺序与 PEFT CPU `merge_and_unload()` 一致：FP32 计算 `B @ A`，将
  delta 转回 BF16，再与 BF16 基础权重相加。
- 每次只处理一个约 5 GiB 的基础分片，使用临时文件和原子重命名。
- 输出保持原来的 282 个分片及索引，完成后是标准 Hugging Face checkpoint。
- 已完成的分片会校验后跳过，可以断点续跑。
- 未修改分片默认复制而非硬链接，保证输出目录独立。

L3 会修改 279/282 个基础分片，因此仍需约 1.37 TiB 的新增磁盘空间。
