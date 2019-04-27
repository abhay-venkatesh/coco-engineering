from pathlib import Path
from pycocotools.coco import COCO
from tqdm import tqdm


class PUBoxBuilder:
    def build(self, split, config):
        if split not in ["train", "val"]:
            raise ValueError("Split " + split + " not supported.")

        split_path = Path(config["destination"], split)

        # Load image ids
        annotations_path = Path(config["source"], "annotations",
                                "stuff_" + split + "2017.json")
        coco = COCO(annotations_path)
        cat_ids = coco.getCatIds(supNms=config["supercategories"])
        img_ids = coco.getImgIds(catIds=cat_ids)
        img_ids = img_ids[:len(img_ids) * config["size fraction"]]

        # Build the dataset
        print("Building " + split + " split...")
        image_path = Path(split_path, "images")
        target_path = Path(split_path, "targets")
        for img_id in tqdm(img_ids):
            pass

        raise NotImplementedError
