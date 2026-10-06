# PhytoRegNet 最小预测发行版

这是供论文审阅使用的独立版本：安装环境、加载真实模型权重、输入 DNA 序列并获得组织分辨的可及性信号预测。英文主页见 [README.md](README.md)。

## 安装

使用 Python **3.10–3.12**。代码、配置、拟南芥和水稻 NIP 各一份真实 fold 0 权重以及合成示例 FASTA 均随安装包提供，安装后即可预测。

当前 `0.1.0` 安装包已在本地准备，尚未上传公共包仓库。发布后可使用：

```bash
python -m pip install phytoregnet
```

Linux x86-64 或 Windows 的 CPU 用户，可以先安装 CPU 版 PyTorch，以减少依赖下载量：

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install phytoregnet
```

conda 包上传后，将 `YOUR_CHANNEL` 替换为发布者的 Anaconda.org channel 名称，在已有的 Python 3.10–3.12 环境中执行：

```bash
conda install --override-channels -c YOUR_CHANNEL -c conda-forge phytoregnet
```

或者新建环境：

```bash
conda create -n phytoregnet --override-channels \
  -c YOUR_CHANNEL -c conda-forge python=3.11 phytoregnet
conda activate phytoregnet
```

conda 包默认使用 CPU 版 PyTorch。构建和上传方法见 [发行说明](PUBLISHING.zh-CN.md)。

### 从 GitHub 源码安装

在本目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python -m pip install .
```

CPU 环境固定为 PyTorch 2.6.0＋NumPy 1.26.4，已在 Linux、Python 3.12.7 的独立环境中跑通两个模型。Windows 使用 `.venv\Scripts\activate` 激活环境。Apple Silicon macOS 安装时将 requirements 命令替换为 `python -m pip install torch==2.6.0 numpy==1.26.4`。

也可以从源码创建 Conda 环境：

```bash
conda env create -f environment.yml
conda activate phytoregnet-minimal
python -m pip install .
```

GPU 用户先安装与驱动匹配的 PyTorch CUDA 版本，预测时指定 `--device cuda:0`，具体安装链接见英文 README。

## 直接运行

无需进入源码目录；在自己的工作目录生成示例 FASTA 并预测：

```bash
phytoregnet-predict --write-example demo.fa
phytoregnet-predict \
  --model arabidopsis \
  --fasta demo.fa \
  --output-prefix outputs/arabidopsis_demo
```

```bash
phytoregnet-predict \
  --model rice_nip \
  --fasta demo.fa \
  --output-prefix outputs/rice_demo
```

`--model` 默认选择安装包内对应的真实 fold 0 权重，每次执行单模型预测。示例包含两条合成的 8192 bp 序列，用于演示接口。换成自己的 FASTA 即可预测；也可用 `python -m phytoregnet_minimal` 替代命令名称。

使用同一模型结构的其他权重时，添加 `--checkpoint /path/to/model.pt`。默认权重会校验 SHA-256，清单见 [manifest.json](src/phytoregnet_minimal/weights/manifest.json)。

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

Python 接口使用同样的默认权重：

```python
from phytoregnet_minimal import Predictor
from phytoregnet_minimal.sequence import read_fasta

ids, sequences = read_fasta("demo.fa")
predictor = Predictor("arabidopsis", device="cpu")
result = predictor.predict(sequences, batch_size=2)
```

## 组织顺序和预测逻辑

拟南芥输出顺序为：root、seedling、seed、inflorescences、fruit、leaf、shoot、flower。

水稻 NIP 的显示顺序为：Callus、Plumule、Apical meristem、Pistil、Stamen、Root、Stem、Leaf。程序保留原训练键 `callus, embryo, meristem, female_reproductive_tissue, male_reproductive_tissue, root, stem, leaf`，另提供显示名称。

预测过程：DNA → A/C/G/T one-hot → 带门控跳跃连接的残差卷积 U-Net → 中央区域裁切和 512-bin 压缩 → RoPE Transformer → profile 与 count 两个输出头。采用 float32、eval 模式和推理模式，按输入方向预测。每次加载一个真实 checkpoint，并严格检查所有权重键和形状。

pip 和 conda 的包名为 `phytoregnet`，Python 导入名为 `phytoregnet_minimal`。源码位于 `src/phytoregnet_minimal/`，权重位于其中的 `weights/`。代码和附带模型权重采用 [MIT 许可证](LICENSE)。
