# GPT-2 蒸馏 + 剪枝压缩实战

这是一个围绕 `GPT-2` 展开的本地压缩实验项目，目标是把“蒸馏 + 剪枝”这条经典模型压缩路线完整跑通，并且尽量做到：

- 本地可运行
- 结构清晰
- 支持单卡与多卡
- 能直接产出可比较的 `PPL` 结果

本文档按照博客分享的写法来组织，重点包括：

1. 项目背景
2. 项目内容
3. 操作细节
4. 结果评估
5. 结果分析

---

## 1. 项目背景

大语言模型虽然效果强，但部署成本高。对于类似 `GPT-2` 这样的自回归语言模型，常见的压缩思路主要有三类：

- 蒸馏：让一个更小的 student 去模仿更大的 teacher
- 剪枝：删除一部分不重要的权重，降低有效参数密度
- 量化：把权重和激活压缩到更低精度

本项目聚焦前两种：

- 先把预训练 `GPT-2` 在本地语料上继续微调，作为 teacher baseline
- 再训练一个 6 层的 student GPT-2
- 用 `CE + KL` 做知识蒸馏
- 再对蒸馏后的 student 做非结构化剪枝
- 最后做短期恢复微调，并用 `PPL` 进行对比

实验语料使用了本地 `WikiText-103`，模型框架使用：

- Python 3.x
- PyTorch
- Hugging Face Transformers

---

## 2. 项目内容

### 2.1 项目目录

```text
/home/qyq/gpt2_compression/
├── prepare_dataset_cache.py
├── train_baseline.py
├── distill_gpt2.py
├── prune_gpt2.py
├── eval_ppl.py
├── clean_results.py
├── data/
│   ├── train.txt
│   ├── valid.txt
│   ├── test.txt
│   └── .cache/
├── utils/
│   ├── convert_wikitext103.py
│   ├── data_utils.py
│   ├── distributed.py
│   ├── hf_utils.py
│   ├── metrics.py
│   └── training.py
└── README.md
```

### 2.2 各脚本作用

`prepare_dataset_cache.py`

- 读取 `train.txt / valid.txt / test.txt`
- 用 GPT-2 tokenizer 分词
- 切成固定长度 token block
- 生成缓存到 `data/.cache/`

`train_baseline.py`

- 加载预训练 GPT-2
- 在本地语料上继续 fine-tuning
- 产出 baseline teacher

`distill_gpt2.py`

- `--mode scratch`：训练 6 层 student，不做蒸馏
- `--mode distill`：训练 6 层 student，使用 teacher 蒸馏

`prune_gpt2.py`

- 对蒸馏后的 student 做全局非结构化剪枝
- 然后进行恢复微调

`eval_ppl.py`

- 读取任意模型目录
- 在验证集上评估 `PPL`

`clean_results.py`

- 清理 `results.csv`
- 确保每种模型只保留最新一行

### 2.3 实验链路

本项目完整实验流程如下：

```text
预训练 GPT-2
    ↓
baseline fine-tuning
    ↓
得到 teacher baseline
    ↓
训练 6 层 student
    ├── scratch: 只用 CE
    └── distill: 用 CE + KL
                ↓
         distilled student
                ↓
         pruning + recovery fine-tune
                ↓
     distilled + pruned student
```

---

## 3. 操作细节

### 3.1 本地环境

本项目默认使用本地 `pixelflow` conda 环境执行：

```bash
/home/qyq/.conda/envs/pixelflow/bin/python
```

本地预训练 GPT-2 权重目录：

```text
/data/lmy_data/LLM/GPT2
```

### 3.2 数据准备

如果本地已经有 `WikiText-103` parquet 分片，可先转换成 txt：

```bash
cd /home/qyq/gpt2_compression

/home/qyq/.conda/envs/pixelflow/bin/python utils/convert_wikitext103.py \
  --source_dir /data/qyq/Wikitext/wikitext-103-v1 \
  --output_dir /home/qyq/gpt2_compression/data \
  --strip_lines
```

转换后得到：

- `data/train.txt`
- `data/valid.txt`
- `data/test.txt`

### 3.3 预处理缓存

为了避免大语料在多卡训练时重复分词，建议先做缓存预处理：

```bash
cd /home/qyq/gpt2_compression

/home/qyq/.conda/envs/pixelflow/bin/python prepare_dataset_cache.py \
  --model_name_or_path /data/lmy_data/LLM/GPT2 \
  --train_file data/train.txt \
  --valid_file data/valid.txt \
  --test_file data/test.txt \
  --cache_dir data/.cache \
  --block_size 128
```

这一步会：

- 读取文本
- 分词
- 切 block
- 缓存为 `.pt` 文件

后续 baseline / scratch / distill / prune 都会直接复用这些缓存。

### 3.4 多卡训练说明

当前项目支持单机多卡 `DistributedDataParallel`，推荐使用：

```bash
torchrun --nproc_per_node=2 ...
```

如果并行启动多个 `torchrun` 任务，需要给每个任务指定不同端口，例如：

```bash
--master_port=29511
--master_port=29512
```

否则会报：

```text
EADDRINUSE: address already in use
```

另外：

- `batch_size` 是每张卡的 batch size
- 全局有效 batch 约等于：

```text
batch_size × GPU数 × grad_accum_steps
```

### 3.5 结果文件与模型保存位置

结果表默认写到：

```text
/home/qyq/gpt2_compression/models/results.csv
```

建议的正式模型保存目录如下：

- baseline teacher：
  - `/data/qyq/GPT2_compression`
- student scratch：
  - `/data/qyq/GPT2_compression_student_scratch`
- student distilled：
  - `/data/qyq/GPT2_compression_student_distilled`
- distilled + pruned：
  - `/data/qyq/GPT2_compression_student_pruned_40`

### 3.6 正式运行命令

以下命令默认使用 2 张 GPU：`2,3`

#### Step 1: baseline

```bash
cd /home/qyq/gpt2_compression

CUDA_VISIBLE_DEVICES=2,3 /home/qyq/.conda/envs/pixelflow/bin/torchrun \
  --master_port=29511 \
  --nproc_per_node=2 \
  train_baseline.py \
  --model_name_or_path /data/lmy_data/LLM/GPT2 \
  --train_file data/train.txt \
  --valid_file data/valid.txt \
  --output_dir /data/qyq/GPT2_compression \
  --results_path /home/qyq/gpt2_compression/models/results.csv \
  --cache_dir data/.cache \
  --epochs 1 \
  --batch_size 4 \
  --eval_batch_size 4 \
  --block_size 128 \
  --grad_accum_steps 2 \
  --learning_rate 5e-5 \
  --num_workers 4 \
  --use_amp
```

#### Step 2: Student scratch

```bash
cd /home/qyq/gpt2_compression

CUDA_VISIBLE_DEVICES=2,3 /home/qyq/.conda/envs/pixelflow/bin/torchrun \
  --master_port=29512 \
  --nproc_per_node=2 \
  distill_gpt2.py \
  --mode scratch \
  --teacher_model_name_or_path /data/qyq/GPT2_compression \
  --output_dir /data/qyq/GPT2_compression_student_scratch \
  --results_path /home/qyq/gpt2_compression/models/results.csv \
  --train_file data/train.txt \
  --valid_file data/valid.txt \
  --cache_dir data/.cache \
  --epochs 3 \
  --batch_size 4 \
  --eval_batch_size 4 \
  --block_size 128 \
  --grad_accum_steps 2 \
  --learning_rate 3e-4 \
  --num_workers 4 \
  --student_init random \
  --use_amp
```

#### Step 3: Student distilled

```bash
cd /home/qyq/gpt2_compression

CUDA_VISIBLE_DEVICES=2,3 /home/qyq/.conda/envs/pixelflow/bin/torchrun \
  --master_port=29513 \
  --nproc_per_node=2 \
  distill_gpt2.py \
  --mode distill \
  --teacher_model_name_or_path /data/qyq/GPT2_compression \
  --output_dir /data/qyq/GPT2_compression_student_distilled \
  --results_path /home/qyq/gpt2_compression/models/results.csv \
  --train_file data/train.txt \
  --valid_file data/valid.txt \
  --cache_dir data/.cache \
  --epochs 2 \
  --batch_size 4 \
  --eval_batch_size 4 \
  --block_size 128 \
  --grad_accum_steps 2 \
  --learning_rate 3e-4 \
  --num_workers 4 \
  --alpha 0.5 \
  --temperature 2.0 \
  --student_init teacher_copy \
  --use_amp
```

#### Step 4: Distilled + Pruned

```bash
cd /home/qyq/gpt2_compression

CUDA_VISIBLE_DEVICES=2,3 /home/qyq/.conda/envs/pixelflow/bin/torchrun \
  --master_port=29514 \
  --nproc_per_node=2 \
  prune_gpt2.py \
  --model_path /data/qyq/GPT2_compression_student_distilled \
  --output_dir /data/qyq/GPT2_compression_student_pruned_40 \
  --results_path /home/qyq/gpt2_compression/models/results.csv \
  --train_file data/train.txt \
  --valid_file data/valid.txt \
  --cache_dir data/.cache \
  --sparsity 0.4 \
  --finetune_epochs 1 \
  --batch_size 4 \
  --eval_batch_size 4 \
  --block_size 128 \
  --grad_accum_steps 2 \
  --learning_rate 1e-4 \
  --num_workers 4 \
  --use_amp
```

#### Step 5: 单独评估任意模型

```bash
cd /home/qyq/gpt2_compression

CUDA_VISIBLE_DEVICES=2,3 /home/qyq/.conda/envs/pixelflow/bin/torchrun \
  --master_port=29515 \
  --nproc_per_node=2 \
  eval_ppl.py \
  --model_path /data/qyq/GPT2_compression_student_distilled \
  --valid_file data/valid.txt \
  --cache_dir data/.cache \
  --block_size 128 \
  --eval_batch_size 4
```

---

## 4. 结果评估

### 4.1 评价指标

本项目使用 `Perplexity (PPL)` 作为核心评价指标：

```text
PPL = exp(loss)
```

对于语言模型任务：

- PPL 越低越好
- 表示模型对下一个 token 的预测越准确

### 4.2 各模型的意义

`GPT2 baseline`

- 预训练 GPT-2 在本地语料上的 fine-tuned teacher

`Student scratch`

- 6 层 student
- 随机初始化
- 不使用 teacher 蒸馏信号

`Student distilled`

- 6 层 student
- 使用 teacher 的 logits 做蒸馏
- loss 为 `CE + KL`

`Distilled + Pruned`

- 先得到蒸馏 student
- 再做剪枝
- 再做恢复微调

### 4.3 当前样例结果

当前本地 `results.csv` 中已有的样例结果如下：

| Model | Params | Sparsity | PPL |
| --- | --- | --- | --- |
| GPT2 baseline | 124439808 | 0.0% | 21.4467 |
| Student scratch | 81912576 | 0.0% | 22214.4909 |
| Student distilled | 81912576 | 0.0% | 24.9270 |
| Distilled + Pruned | 81912576 | 40.0% | 20293.4342 |

但要注意，这张表目前并不完全是同一轮正式实验的最终结果：

- `baseline` 和 `distilled` 已经是比较可信的新结果
- `scratch` 和 `pruned` 仍然可能是旧 smoke-test 结果
- 因此需要按上面的完整命令重新正式跑一遍，才能得到真正可比的 4 行结果

---

## 5. 结果分析

### 5.1 baseline 的意义

`baseline` 不是“直接拿原始 GPT-2 做评估”，而是：

- 先加载预训练 GPT-2
- 再在本地语料上 fine-tune
- 最后在验证集上计算 PPL

因此 `baseline` 可以理解为：

- 当前数据分布上的 teacher 模型

### 5.2 scratch 与 distilled 的区别

`scratch`：

- student 从随机初始化开始
- 只用真实标签做监督
- loss 只有 `CE`

`distilled`：

- student 同时学习真实标签和 teacher 输出分布
- loss 为：

```text
L = alpha * CE + (1 - alpha) * KL
```

其中：

- `alpha = 0.5`
- `temperature = 2.0`
- KL 项乘了 `T^2`

如果蒸馏有效，通常会看到：

- `Student distilled` 明显优于 `Student scratch`

### 5.3 为什么当前 `scratch` 和 `pruned` 看起来很差

从目前现象判断，更大的可能不是：

- 压缩方法完全失效

而是：

- `Student scratch` 还没有用正式配置重新完整跑完
- `Distilled + Pruned` 也还没有基于最新 distilled student 正式跑完

因此当前合理的解读是：

- baseline 正常
- distilled 正常，并且已经证明蒸馏有效
- scratch / pruned 还需要重新正式运行

### 5.4 对项目落地的理解

这个项目最有价值的点，不只是把一个结果表跑出来，而是把完整压缩路线真正工程化：

- 数据转换
- 预处理缓存
- 单卡 / 多卡训练
- baseline / scratch / distill / prune 的统一结果记录
- 模型目录与评估流程统一管理

这让整个实验不再是一次性脚本，而是一条可复现实验流水线。

---

## 6. 常见问题

### 6.1 `HFValidationError: Repo id must be in the form ...`

通常说明：

- 你传给 `--teacher_model_name_or_path` 或 `--model_path` 的本地目录不存在

例如：

```text
/data/qyq/GPT2_compression
```

如果这个目录不存在，`transformers` 会把它误当成 Hugging Face repo id 去解析。

解决办法：

- 先确认 baseline 是否真的保存成功
- 再运行：

```bash
ls -la /data/qyq/GPT2_compression
```

### 6.2 `EADDRINUSE: address already in use`

说明两个 `torchrun` 任务使用了同一个 `master_port`。  
解决方法：

- 给每个任务设置不同的 `--master_port`

### 6.3 多卡缓存读写报错 / `EOFError`

之前遇到过缓存文件未完整写完时被其他 rank 读取的问题。现在代码已经改成：

- 主进程先写临时文件
- 写完后原子重命名
- 其他进程等待完整缓存出现后再加载

### 6.4 结果表里出现多个同名模型

项目现在已经支持：

- 同名模型覆盖旧结果

如果还需要手动清理，可以运行：

```bash
cd /home/qyq/gpt2_compression
/home/qyq/.conda/envs/pixelflow/bin/python clean_results.py
```

---

## 7. 总结

这个项目实现了一条完整的 GPT-2 压缩实验链：

- baseline teacher fine-tuning
- student scratch 训练
- teacher-student 蒸馏
- 剪枝与恢复微调
- 使用统一的 PPL 指标做结果比较

从当前结果看：

- baseline 已经稳定
- distilled 已经接近 baseline
- scratch 和 pruned 还需要基于最新正式模型再跑一轮

从工程角度看，这个项目已经具备博客分享、实验复现和后续扩展的基础。后续如果继续扩展，可以考虑：

- 加入量化实验
- 加入 teacher-layer 更灵活的映射策略
- 支持更长 context
- 增加 test set 最终评估
- 加入 WandB / TensorBoard 等更完整日志系统
