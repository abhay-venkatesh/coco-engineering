# coco-stuff-tools

Engineering the MS COCO Stuff dataset.

## Setup

### Ubuntu

```bash
conda env create
```

### Windows

pycocotools is not available on conda for windows. Hence, do

```bash
conda env create -f windows_env.yml
conda install cython
pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
```
