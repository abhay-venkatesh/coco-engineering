# coco-stuff-tools

Engineering the MS COCO Stuff dataset.

## Setup

### Ubuntu

```bash
conda env create -f ubuntu_env.yml
pip install pycocotools
```

### Windows

```bash
conda env create -f windows_env.yml
pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
```
