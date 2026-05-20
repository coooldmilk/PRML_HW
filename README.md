# HW4 Transformer 复现与位置编码消融实验

HW4包含一份实验报告和两份代码，用于完成 Transformer 复现、合理性检验以及 IWSLT 英德翻译任务上的位置编码消融实验。

## 1. 文件说明

| 文件 | 说明 |
|---|---|
| `HW4.pdf` | 实验报告，包含 Transformer 复现思路、IWSLT 2017 英德翻译实验、位置编码消融实验和结果分析。 |
| `Transformer_sanity_check.py` | 代码 1：合理性检验代码。在合成的反转序列任务上验证 Transformer 编码器—解码器结构是否能正常训练和生成。 |
| `Transformer_IWSLT.py` | 代码 2：真实机器翻译实验代码。在 IWSLT 2017 English-German 数据集上训练小型 Transformer，并输出验证损失、词级准确率、翻译样例和 BLEU。 |

## 2. 推荐环境

本实验已在以下环境中跑通：

```text
操作系统：Windows 10
Python：3.11.7
PyTorch：2.5.1+cu121
CUDA：12.1
cuDNN：90100
GPU：NVIDIA GeForce RTX 4060 Laptop GPU
Matplotlib：3.10.7
NumPy：1.26.4
datasets：2.20.0
```

由于当前 IWSLT 数据集在部分最新版 `datasets` 环境中可能无法正常加载，建议将 `datasets` 降到本实验使用的版本：

```bash
pip install datasets==2.20.0
```

其他依赖可安装为：

```bash
pip install torch tokenizers tqdm sacrebleu matplotlib numpy
```


## 3. 数据集说明

代码 2 使用 Hugging Face 上的 IWSLT 2017 英德翻译数据集：

```text
https://huggingface.co/datasets/IWSLT/iwslt2017
```

本实验使用的数据集配置为：

```text
dataset name: IWSLT/iwslt2017
dataset config: iwslt2017-en-de
source language: en
target language: de
```

数据集可以自行从 Hugging Face 下载，也可以直接运行代码，由程序自动下载并缓存到本地。

第一次运行请联网！！！由于该数据集需要加载远程脚本，运行时需要加入：

```bash
--trust-remote-code
```

可先用下面的命令测试数据集是否能正常加载：

```bash
python - <<'PY'
from datasets import load_dataset

ds = load_dataset(
    "IWSLT/iwslt2017",
    "iwslt2017-en-de",
    split="train[:1]",
    trust_remote_code=True
)
print(ds[0])
PY
```

如果能输出一条包含 `translation` 字段的英德句对，说明数据集加载正常。

## 4. 代码 1 使用方法：合理性检验

代码文件：

```text
Transformer_sanity_check.py
```

该代码用于在合成的反转序列任务上检查 Transformer 是否能正常工作。例如输入序列为：

```text
[7, 4, 9, 5]
```

目标输出为：

```text
[5, 9, 4, 7]
```

该任务不用于真实翻译，只用于验证模型结构是否实现正确。

运行方式很简单，直接执行：

```bash
python Transformer_sanity_check.py
```

程序会自动生成合成数据，训练模型，并输出训练损失、验证损失和词级准确率。训练结束后，还会打印若干条预测样例，用于观察模型是否学会反转序列。

如需调整训练轮数或模型规模，可以使用参数，例如：

```bash
python Transformer_sanity_check.py --epochs 20 --d-model 128 --num-layers 3
```

一般情况下，代码 1 直接运行即可，不需要修改参数。

## 5. 代码 2 使用方法：IWSLT 英德翻译实验

代码文件：

```text
Transformer_IWSLT.py
```

该代码用于真实机器翻译实验，整体流程如下：

1. 自动加载 IWSLT 2017 English-German 数据集。
2. 使用训练集构建英德共享 BPE tokenizer。
3. 构建小型 Transformer 编码器—解码器模型。
4. 使用 Adam、Noam learning rate schedule 和 label smoothing 训练模型。
5. 使用 beam search 进行翻译解码。
6. 输出验证损失、词级准确率、翻译样例和 BLEU。

### 5.1 Windows PowerShell 运行命令

在 Windows PowerShell 中运行：

```powershell
python Transformer_IWSLT.py `
  --dataset-name IWSLT/iwslt2017 `
  --dataset-config iwslt2017-en-de `
  --src-lang en `
  --tgt-lang de `
  --trust-remote-code `
  --train-limit 50000 `
  --valid-limit 888 `
  --vocab-size 16000 `
  --retrain-tokenizer `
  --output-dir runs\iwslt_50k_d256 `
  --d-model 256 `
  --num-layers 4 `
  --num-heads 4 `
  --d-ff 1024 `
  --dropout 0.1 `
  --label-smoothing 0.1 `
  --warmup-steps 4000 `
  --epochs 30 `
  --batch-size 64 `
  --num-workers 0 `
  --eval-bleu-every 0 `
  --decode-strategy beam `
  --beam-size 4 `
  --length-penalty-alpha 0.6
```

### 5.2 主要参数说明

| 参数 | 含义 |
|---|---|
| `--dataset-name` | Hugging Face 数据集名称。 |
| `--dataset-config` | 数据集配置，这里使用 `iwslt2017-en-de`。 |
| `--src-lang` | 源语言，本文为英文 `en`。 |
| `--tgt-lang` | 目标语言，本文为德文 `de`。 |
| `--trust-remote-code` | 允许加载 Hugging Face 数据集脚本。 |
| `--train-limit` | 使用的训练样本数，本文为 50000。 |
| `--valid-limit` | 使用的验证样本数，本文为 888。 |
| `--vocab-size` | BPE 词表大小，本文为 16000。 |
| `--retrain-tokenizer` | 重新训练 tokenizer。 |
| `--output-dir` | 模型、tokenizer 和配置文件保存目录。 |
| `--d-model` | Transformer 隐藏层维度。 |
| `--num-layers` | 编码器和解码器层数。 |
| `--num-heads` | 多头注意力头数。 |
| `--d-ff` | 前馈网络中间层维度。 |
| `--dropout` | Dropout 比例。 |
| `--label-smoothing` | 标签平滑系数。 |
| `--warmup-steps` | Noam 学习率预热步数。 |
| `--epochs` | 训练轮数。 |
| `--batch-size` | 批大小。 |
| `--num-workers` | DataLoader 进程数；Windows 下建议使用 0。 |
| `--eval-bleu-every` | 训练过程中每隔多少轮计算 BLEU；设为 0 表示训练中不额外计算。 |
| `--decode-strategy` | 解码方式，可选 `greedy` 或 `beam`。 |
| `--beam-size` | beam search 束宽。 |
| `--length-penalty-alpha` | beam search 长度惩罚系数。 |

## 6. 输出结果说明

运行代码 2 后，会在 `--output-dir` 指定目录下保存结果，例如：

```text
runs/iwslt_50k_d256/
```

主要输出文件包括：

| 文件 | 说明 |
|---|---|
| `config.json` | 保存本次运行参数。 |
| `shared_bpe_tokenizer.json` | 训练得到的英德共享 BPE tokenizer。 |
| `last.pt` | 最后一轮模型 checkpoint。 |
| `best.pt` | 验证损失最低的模型 checkpoint。 |

训练过程中会输出每轮的训练损失、验证损失和词级准确率。训练结束后，程序会加载 `best.pt`，打印若干条翻译样例，并计算最终 BLEU。

## 7. 推荐运行顺序

建议先运行代码 1，确认 Transformer 基础结构无误：

```bash
python Transformer_sanity_check.py
```

确认合理性检验正常后，再运行代码 2：

```bash
python Transformer_IWSLT.py --dataset-name IWSLT/iwslt2017 --dataset-config iwslt2017-en-de --src-lang en --tgt-lang de --trust-remote-code --train-limit 50000 --valid-limit 888 --vocab-size 16000 --retrain-tokenizer --output-dir runs/iwslt_50k_d256 --d-model 256 --num-layers 4 --num-heads 4 --d-ff 1024 --dropout 0.1 --label-smoothing 0.1 --warmup-steps 4000 --epochs 30 --batch-size 64 --num-workers 0 --eval-bleu-every 0 --decode-strategy beam --beam-size 4 --length-penalty-alpha 0.6
```

## 8. 一些遇到过的问题，请留意！！！

### 数据集加载失败

优先检查 `datasets` 版本，推荐使用：

```bash
pip install datasets==2.20.0
```

并确认运行命令中包含：

```bash
--trust-remote-code
```

### Windows 下 DataLoader 报错

Windows 环境建议设置：

```bash
--num-workers 0
```

### 训练速度较慢或显存不足

可以适当减小以下参数：

```bash
--batch-size
--d-model
--num-layers
--d-ff
```

例如将 batch size 改为 32：

```bash
--batch-size 32
```

## 9. 最后一句话

如有任何报错，请直接询问GPT，代码本身可以跑通，大概率是环境或者版本问题，很好解决！！！
