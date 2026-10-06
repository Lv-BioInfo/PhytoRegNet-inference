# PhytoRegNet 包构建和发布

这是维护者使用的发行说明。包名为 `phytoregnet`，初始版本为 `0.1.0`，Python 导入名为 `phytoregnet_minimal`，预测命令为 `phytoregnet-predict`。代码和包内两份 fold 0 模型权重均采用 MIT 许可证。

当前阶段为本地构建和验证，公共 PyPI 与 Anaconda.org 上传尚未执行。GitHub 推送、PyPI 发布和 conda channel 上传是三个独立操作。

## 构建 pip 包

在项目根目录中，使用单独的构建环境：

```bash
python3 -m venv .build-venv
source .build-venv/bin/activate
python -m pip install --upgrade build twine
python -m build
python -m twine check --strict dist/phytoregnet-0.1.0-py3-none-any.whl dist/phytoregnet-0.1.0.tar.gz
```

产物为：

- `dist/phytoregnet-0.1.0-py3-none-any.whl`：可直接安装的 wheel。
- `dist/phytoregnet-0.1.0.tar.gz`：含同一份权重的源码发行包。

运行时只依赖 NumPy 和 PyTorch；构建与上传工具不作为运行时依赖。权重、配置和示例 FASTA 都封装在 `phytoregnet_minimal` 包内。每个文件小于 PyPI 默认的 100 MB 上传上限。

在另一个 Python 3.10–3.12 环境测试 wheel，例如 Linux x86-64 CPU 环境：

```bash
python3 -m venv .test-venv
source .test-venv/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install dist/phytoregnet-0.1.0-py3-none-any.whl
python -m pip check
phytoregnet-predict --write-example demo.fa
phytoregnet-predict --model arabidopsis --fasta demo.fa --output-prefix outputs/arabidopsis
phytoregnet-predict --model rice_nip --fasta demo.fa --output-prefix outputs/rice_nip
```

正式发布前，将两个 README 中公共包仓库上传待完成的状态文字更新为实际发布状态，重新构建并检查产物。

## 上传 PyPI

需要你持有 PyPI 账号并完成 PyPI 的认证要求；使用该账号的 API token，或另行配置 GitHub Trusted Publishing。当前项目先采用 Twine 交互式上传，无需在源码中保存 token。

在装有 Twine 的构建环境执行：

```bash
python -m twine upload dist/phytoregnet-0.1.0-py3-none-any.whl dist/phytoregnet-0.1.0.tar.gz
```

按提示输入用户名 `__token__`，密码位置输入 PyPI API token。新项目的第一次上传需要具有创建项目权限的凭据。不要将 token 写入本仓库。

如需先检查公共上传流程，可以使用独立的 TestPyPI 账号和 token：

```bash
python -m twine upload --repository testpypi dist/phytoregnet-0.1.0-py3-none-any.whl dist/phytoregnet-0.1.0.tar.gz
```

上传成功后，在新环境中确认：

```bash
python -m pip install phytoregnet==0.1.0
phytoregnet-predict --version
```

PyPI 包名尚未显示为公开项目不代表一定可注册，最终以第一次上传的结果为准。已经发布的同版本发行文件不能覆盖；修改后需要增加版本号。

## 构建 conda 包

在单独的构建环境中安装构建工具：

```bash
conda create -n phytoregnet-build --override-channels -c conda-forge conda-build anaconda-client
conda activate phytoregnet-build
mkdir -p dist/conda
conda build conda-recipe --override-channels -c conda-forge \
  --python 3.11 --no-anaconda-upload --output-folder dist/conda
```

配方位于 [conda-recipe/meta.yaml](conda-recipe/meta.yaml)。构建产生 `noarch: python` 包，使用 conda-forge 的 `pytorch-cpu`，包括与 wheel 相同的权重和配置。构建时自动生成包内示例 FASTA，并运行两种模型的预测。

从本地 channel 安装：

```bash
conda create -n phytoregnet-local --override-channels \
  -c ./dist/conda -c conda-forge python=3.11 phytoregnet
conda activate phytoregnet-local
phytoregnet-predict --write-example demo.fa
phytoregnet-predict --model arabidopsis --fasta demo.fa --output-prefix outputs/local_demo
```

## 上传 conda channel

需要 Anaconda.org 账号及可上传的个人或组织 channel。先通过交互式登录，再上传实际生成的 conda 包：

```bash
anaconda login
anaconda upload --user YOUR_CHANNEL dist/conda/noarch/phytoregnet-0.1.0-py_0.tar.bz2
```

将 `YOUR_CHANNEL` 替换为自己的 Anaconda.org 用户名或有权限的组织名，它不一定与 GitHub 用户名相同。本次本地构建使用 `.tar.bz2` 格式；如其他构建工具生成 `.conda`，使用实际文件名上传。

上传后，用户在已有的 Python 3.10–3.12 环境中运行：

```bash
conda install --override-channels -c YOUR_CHANNEL -c conda-forge phytoregnet
```

自己的 channel 已足够用于论文审稿；加入 conda-forge 或 Bioconda 是后续可选的维护工作。

## 更新版本

同步修改 `pyproject.toml`、`src/phytoregnet_minimal/__init__.py` 和 `conda-recipe/meta.yaml` 中的版本号。清理旧的本地发行产物后重新构建、验证和上传。若 conda 仅修改依赖或配方而不改源码版本，增加配方的 `build.number`。

对应 GitHub Release 使用相同版本的标签，例如 `v0.1.0`。完成公共上传后在 README 中填入实际 channel 并更新安装状态。

官方依据：[Python 包构建与发布](https://packaging.python.org/en/latest/tutorials/packaging-projects/)、[PyPI 文件限制](https://docs.pypi.org/project-management/storage-limits/)、[Twine 使用说明](https://twine.readthedocs.io/en/stable/)、[conda 包构建](https://docs.conda.io/projects/conda-build/en/stable/user-guide/tutorials/build-pkgs.html)、[Anaconda.org 包上传](https://www.anaconda.com/docs/tools/anaconda-org/maintainer-guide/labels)。
