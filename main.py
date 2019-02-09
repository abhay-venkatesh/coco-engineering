from datasets.coco_stuff_weak_bb import build_coco_stuff_weak_bb
from lib.config import get_config

if __name__ == "__main__":
    config = get_config()
    build_coco_stuff_weak_bb(config)
