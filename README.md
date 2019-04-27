# coco-stuff-tools

Engineering the MS COCO Stuff dataset.

## Setup

### Ubuntu (Coming Soon)

```bash
conda env create -f ubuntu_env.yml
```

### Windows

pycocotools is not available on conda for windows. Hence, do

```bash
conda env create -f windows_env.yml
pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
```
