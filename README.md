# coco-stuff-tools

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

## Running

```bash
python main.py /path/to/config.yml
```
