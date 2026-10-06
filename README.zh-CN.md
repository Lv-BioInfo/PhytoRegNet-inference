# PhytoRegNet 最小预测发行版

这是供论文审阅使用的独立版本：安装环境、加载真实模型权重、输入 DNA 序列并获得组织分辨的可及性信号预测。英文主页见 [README.md](README.md)。

## 安装

使用 Python 3.10–3.12。在当前目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python -m pip install .
```

默认 CPU 环境为 PyTorch 2.6.0＋NumPy 1.26.4，已在 Linux、Python 3.12.7 的独立环境中安装并跑通两个示例。Windows 使用 `.venv\Scripts\activate` 激活环境。

Apple Silicon macOS 安装时将 requirements 命令替换为 `python -m pip install torch==2.6.0 numpy==1.26.4`，然后安装本包。

也可以使用 Conda：

```bash
conda env create -f environment.yml
conda activate phytoregnet-minimal
python -m pip install .
```

GPU 安装方式见英文 README；选择与驱动匹配的 PyTorch CUDA 版本，预测时指定 `--device cuda:0`。

## 直接运行

附带拟南芥和水稻 NIP 各一个论文原始 fold 0 权重。下面的命令分别执行单模型预测：

```bash
phytoregnet-predict \
  --model arabidopsis \
  --checkpoint weights/arabidopsis/fold_0.pt \
  --fasta examples/demo.fa \
  --output-prefix outputs/arabidopsis_demo
```

```bash
phytoregnet-predict \
  --model rice_nip \
  --checkpoint weights/rice_nip/fold_0.pt \
  --fasta examples/demo.fa \
  --output-prefix outputs/rice_demo
```

示例 FASTA 是两条合成的 8192 bp 序列，用于演示接口。换成自己的 FASTA 即可预测；也可用 `python -m phytoregnet_minimal` 替代命令名称。

## 输入和输出

每条输入必须恰好 **8192 bp**；支持 A/C/G/T/N 和大小写，N 编码为四个零。支持多行 FASTA，每条记录的第一个标题字段作为唯一 ID。长度错误、重复 ID、空记录和其他字符会报错。

模型预测输入中央 **4096 bp**，即相对每条 FASTA 记录的 `[2048,6144)`，分为 **512 个 8 bp bins**。第 `i` 个 bin 为 `[2048+8i,2056+8i)`，坐标零基、左闭右开。

| 输出文件 | 内容 |
| --- | --- |
| `PREFIX.npz` | 完整 profile、log_count、count、组织顺序、序列 ID 和 bin 坐标 |
| `PREFIX.counts.tsv` | 每条序列、每个组织的原始 log_count 及还原后的 count |
| `PREFIX.metadata.json` | 模型参数、权重校验值和预测设置 |

添加 `--profile-tsv` 可另输出逐 bin 的信号表。使用 `--batch-size` 调整批大小、`--threads` 调整 CPU 线程数。已有结果默认保留，用 `--overwrite` 才会覆盖。

```python
import numpy as np

with np.load("outputs/arabidopsis_demo.npz", allow_pickle=False) as result:
    print(result["track_names"])
    print(result["profile"].shape)    # (2, 8, 512)：序列、组织、bin
    print(result["log_count"].shape)  # (2, 8)：序列、组织
```

`profile` 是训练归一化信号尺度上的非负预测，由 Softplus 输出。`log_count` 由独立输出头预测中央区域的 `log(1＋归一化总信号)`；`count=max(expm1(log_count),0)`。两个输出头各自预测，profile 的总和不要求等于 count。原始 log_count 即使为负也会保留。

## 组织顺序和预测逻辑

拟南芥输出顺序为：root、seedling、seed、inflorescences、fruit、leaf、shoot、flower。

水稻 NIP 的显示顺序为：Callus、Plumule、Apical meristem、Pistil、Stamen、Root、Stem、Leaf。程序保留原训练键 `callus, embryo, meristem, female_reproductive_tissue, male_reproductive_tissue, root, stem, leaf`，另提供显示名称，不改变输出顺序。

预测过程：DNA → A/C/G/T one-hot → 带门控跳跃连接的残差卷积 U-Net → 中央区域裁切和 512-bin 压缩 → RoPE Transformer → profile 与 count 两个输出头。采用 float32、eval 模式和推理模式，按输入方向预测。每次加载一个真实 checkpoint，并严格检查所有权重键和形状。

安装包入口、模型结构及加载逻辑位于 `src/phytoregnet_minimal/`。权重在 `weights/`，校验值见 [manifest.json](weights/manifest.json)；推理配置和组织顺序随 Python 包安装。发行时将本目录整体作为独立的 GitHub 审阅版本管理。
