from PIL import Image
from lib.box_builder import BoxBuilder
from pathlib import Path
from pycocotools.coco import COCO
from tqdm import tqdm
import csv
import os
import shutil


class Builder:
    def __init__(self, paths, config):
        self.paths = paths
        self.config = config

        if not os.path.exists(self.paths["filtered_data_root"]):
            os.makedirs(self.paths["filtered_data_root"])

        self.dataset_root = Path(self.paths["filtered_data_root"],
                                 self.config["name"])
        if os.path.exists(self.dataset_root):
            shutil.rmtree(self.dataset_root)
        os.makedirs(self.dataset_root)

        for split in self.config["splits"]:
            filtered_split_folder = Path(self.dataset_root, split["name"])
            if not os.path.exists(filtered_split_folder):
                os.mkdir(filtered_split_folder)

        self.target_supercategories = config["target supercategories"]
        self.box_type = config["box type"]
        self.n_boxes = config["number of boxes"]

    def _filter_annotations_file(self, coco, img_ids, target_cat_ids):
        filtered_ann_path = Path(self.filtered_data_location,
                                 "annotations.csv")
        with open(
                filtered_ann_path, mode='w', newline='') as filtered_ann_file:
            writer = csv.writer(
                filtered_ann_file,
                delimiter=',',
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL)
            for img_id in img_ids:
                ann_ids = coco.getAnnIds(imgIds=img_id)
                anns = coco.loadAnns(ann_ids)
                for ann in anns:
                    if ann["category_id"] in target_cat_ids:
                        for i in range(self.n_boxes):
                            img_name = coco.loadImgs(img_id)[0]['file_name']
                            writer.writerow([img_name, i])

    def _filter_dataset(self,
                        ann_file_path,
                        data_root,
                        filtered_data_location,
                        fraction=1.0):
        coco = COCO(ann_file_path)
        box_builder = BoxBuilder(self.box_type, coco, filtered_data_location)
        img_ids = coco.getImgIds()
        img_ids = img_ids[:int(fraction * len(img_ids))]
        target_cat_ids = coco.getCatIds(supNms=self.target_supercategories)

        for img_id in tqdm(img_ids):
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            for ann in anns:
                if ann["category_id"] in target_cat_ids:
                    img_name = coco.loadImgs(img_id)[0]['file_name']
                    img_path = Path(data_root, img_name)
                    img_path_ = Path(filtered_data_location, "images",
                                     img_name)
                    if not os.path.exists(
                            Path(filtered_data_location, "images")):
                        os.makedirs(Path(filtered_data_location, "images"))
                    shutil.copyfile(img_path, img_path_)

                    seg_array = coco.annToMask(ann)
                    seg = Image.fromarray(seg_array)
                    seg_name = coco.loadImgs(img_id)[0]['file_name'].replace(
                        ".jpg", ".png")
                    seg_path = Path(filtered_data_location, "annotations",
                                    seg_name)
                    if not os.path.exists(
                            Path(filtered_data_location, "annotations")):
                        os.makedirs(
                            Path(filtered_data_location, "annotations/"))
                    seg.save(seg_path)

                    box_builder.build(img_id, ann, self.n_boxes)

        self._filter_annotations_file(img_ids, target_cat_ids)

    def build(self):
        self.config = self.config
        self.paths = self.paths

        for split in self.config["splits"]:
            filtered_split_folder = Path(self.dataset_root, split["name"])
            ann_file_name = split["name"] + "_ann_file"
            root_name = split["name"] + "_root"
            self._filter_dataset(self.paths[ann_file_name],
                                 self.paths[root_name], filtered_split_folder,
                                 split["fraction"])
