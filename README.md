# gcn-example
Example graph convolutional network implementation with the PyTorch Geometric library

## Installation on Mac
### PyTorch 1.0.0 or later:
```bash
conda install pytorch torchvision -c pytorch
```
### PyTorch Geometric
```bash
pip install --verbose --no-cache-dir torch-scatter
pip install --verbose --no-cache-dir torch-sparse
pip install --verbose --no-cache-dir torch-cluster
pip install --verbose --no-cache-dir torch-spline-conv (optional)
pip install torch-geometric
```
On my system, torch-scatter and torch-cluster failed to install throwing the following error:
```bash
                                   !! WARNING !!

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    Your compiler (g++) is not compatible with the compiler Pytorch was
    built with for this platform, which is clang++ on darwin. Please
    use clang++ to to compile your extension. Alternatively, you may
    compile PyTorch from source using g++, and then you can also use
    g++ to compile your extension.

    See https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md for help
    with compiling PyTorch from source.
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                                  !! WARNING !!
```
```bash
'gcc' failed with exit status 1
```
Solution from https://github.com/rusty1s/pytorch_scatter/issues/21:
```bash
MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ pip install torch-scatter
MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ pip install torch-cluster
```
Also, update to the most recent version of PyTorch if problems persist
