# data-engineering

Engineering data for machine learning. 

## Organization

If single dataset:

```
lib/
cache/
configs/
stats/
main.py
```

else:
```
datasets/
    dataset_1/
        lib/
        cache/
        configs/
        stats/
    dataset_2/
        lib/
        cache/
        configs/
        stats/
    ...
main.py
```

## Usage

Add paths into the lib/coco/paths.py file.