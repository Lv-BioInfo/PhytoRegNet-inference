# Build the conda package

This recipe builds the same inference code and bundled checkpoints as the
Python wheel. It creates a `noarch: python` package; PyTorch supplies the native
CPU libraries for the target platform through `conda-forge`.

From the repository root, with `conda-build` installed in a build environment:

```bash
mkdir -p dist/conda
conda build conda-recipe --override-channels -c conda-forge \
  --python 3.11 --no-anaconda-upload --output-folder dist/conda
```

The build tests the installed command on the example FASTA with both bundled
models. To install the resulting package in a new environment:

```bash
conda create -n phytoregnet-local --override-channels \
  -c ./dist/conda -c conda-forge python=3.11 phytoregnet
conda activate phytoregnet-local
phytoregnet-predict --help
```

Public installation requires uploading the built package to an Anaconda.org
channel. After publication, replace `YOUR_CHANNEL` with that channel name:

```bash
conda create -n phytoregnet --override-channels \
  -c YOUR_CHANNEL -c conda-forge python=3.11 phytoregnet
```

Publishing to a personal or organization channel does not require admission to
conda-forge or Bioconda. The recipe uses `pytorch-cpu` so a standard installation
uses CPU libraries.

References: [conda recipe metadata](https://docs.conda.io/projects/conda-build/en/stable/resources/define-metadata.html),
[building packages](https://docs.conda.io/projects/conda-build/en/stable/user-guide/tutorials/build-pkgs.html),
[conda-forge PyTorch CPU packages](https://github.com/conda-forge/pytorch-cpu-feedstock).
