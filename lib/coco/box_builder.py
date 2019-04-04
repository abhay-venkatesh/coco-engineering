from PIL import Image, ImageDraw
from math import sqrt
from pathlib import Path
import numpy as np
import os
import random


class BoxBuilder:
    def __init__(self, box_type, n_boxes, coco, filtered_data_location,
                 downsample):
        self.box_type = box_type
        self.build = self._filter_bounding_boxes
        if self.box_type == "full":
            self.build = self._filter_full_bounding_boxes
        elif self.box_type == "coordinate":
            self.build = self._filter_coordinate_boxes
        elif type(self.box_type) is dict:
            if self.box_type["name"] == "nonrandom aggregated":
                self.build = self._filted_nonrandom_agg_boxes
            elif self.box_type["name"] == "aggregated":
                self.build = self._filter_agg_bounding_boxes
            elif self.box_type["name"] == "turnedoff":
                self.build = self._filter_turnedoff_mask
            else:
                raise NotImplementedError(
                    "Box type: " + self.box_type["name"] + " not supported.")
        else:
            raise NotImplementedError("Box type: " + self.box_type +
                                      " not supported.")

        self.n_boxes = n_boxes
        assert self.n_boxes >= 1, ("Need to have at least one bounding box. ")
        self.coco = coco

        self.box_location = Path(filtered_data_location, "bbox")
        if not os.path.exists(self.box_location):
            os.makedirs(self.box_location)

        self.downsample = downsample

    def _random_bbox(self, seg_array, smoothing=2):
        """ For example, let
        seg_array = [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        xb, yb, wb, hb = 0, 0, 5, 2 """
        _, _, wb, hb = self._get_seg_boundary(seg_array)
        rows, cols = np.nonzero(seg_array)
        ri = random.randint(0, len(rows) - 1)
        """ Then,
        y in [0, 1]
        x in [0, 1, 2, 3, 4] """
        y, x = rows[ri], cols[ri]
        """ We want
        h in [0, hb - y]
        w in [0, wb - x] """
        h, w = random.randint(0, (hb - y) // smoothing), random.randint(
            0, (wb - x) // smoothing)
        return x, y, w, h

    def _get_seg_boundary(self, seg_array):
        rows, cols = np.nonzero(seg_array)
        x = min(cols)
        w = max(cols)
        y = min(rows)
        h = max(rows)
        return x, y, w, h

    def _draw_random_bbox_from_seg(self, img_id, seg_array):
        img_height = self.coco.loadImgs(img_id)[0]['height']
        img_width = self.coco.loadImgs(img_id)[0]['width']
        seg = Image.fromarray(np.zeros((img_height, img_width)))
        draw = ImageDraw.Draw(seg)
        x, y, w, h = self._random_bbox(seg_array)
        """
        img = [
            [1, 1, 1],
            [0, 0, 0],
            [1, 1, 1],
        ]

        then (x, y) indexes of each element are given by [
            [(0, 0), (0, 1), (0, 2)],
            [(1, 0), (1, 1), (1, 2)],
            [(2, 0), (2, 1), (2, 2)],
        ]
        """
        rect = self._get_rect(x, y, w, h, 0)
        draw.polygon([tuple(p) for p in rect], fill=1)
        np_seg = np.asarray(seg, dtype=int)
        return np_seg

    def _get_rect(self, x, y, width, height, angle):
        """ Get a rectangle from (x, y) and (width, height) """
        rect = np.array([(0, 0), (width, 0), (width, height), (0, height),
                         (0, 0)])
        theta = (np.pi / 180.0) * angle
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        offset = np.array([x, y])
        transformed_rect = np.dot(rect, R) + offset
        return transformed_rect

    def _preview_image(self, img_id, data_root):
        img_name = self.coco.loadImgs(img_id)[0]['file_name']
        img_path = Path(data_root, img_name)
        img = Image.open(img_path)
        img.show()
        input("Press Enter to continue...")

    def _preview_mask(self, seg):
        seg_arr = np.asarray(seg)
        seg_arr = np.multiply(seg_arr, 100)
        mask = Image.fromarray(seg_arr)
        mask.show()
        input("Press Enter to continue...")

    def _filter_bounding_boxes(self, img_id, ann, n_boxes=10):
        seg_array = self.coco.annToMask(ann)

        for i in range(self.n_boxes):
            bbox = self._draw_random_bbox_from_seg(img_id, seg_array)
            bbox = Image.fromarray(bbox).convert("L")
            bbox_name = self.coco.loadImgs(img_id)[0]['file_name'].replace(
                ".jpg", "-" + str(i) + ".png")
            bbox_path = Path(self.box_location, bbox_name)
            bbox.save(bbox_path)

    def _filter_agg_bounding_boxes(self, img_id, ann):
        seg_array = self.coco.annToMask(ann)
        bbox = self._draw_random_bbox_from_seg(img_id, seg_array)
        for i in range(1, self.box_type["num miniboxes"]):
            bbox_ = bbox | self._draw_random_bbox_from_seg(img_id, seg_array)
            bbox = bbox_

        bbox = np.array(bbox, dtype=np.uint8)
        bbox = Image.fromarray(bbox).convert("L")
        bbox_name = self.coco.loadImgs(img_id)[0]['file_name'].replace(
            ".jpg", "-0.png")
        bbox_path = Path(self.box_location, bbox_name)
        bbox.save(bbox_path)

    def _filter_full_bounding_boxes(self, img_id, ann):
        img_height = self.coco.loadImgs(img_id)[0]['height']
        img_width = self.coco.loadImgs(img_id)[0]['width']
        seg = Image.fromarray(np.zeros((img_height, img_width)))
        draw = ImageDraw.Draw(seg)
        x, y, w, h = ann["bbox"]
        rect = self._get_rect(x, y, w, h, 0)
        draw.polygon([tuple(p) for p in rect], fill=1)
        bbox = np.asarray(seg, dtype=np.uint8)
        bbox = Image.fromarray(bbox).convert("L")
        bbox_name = self.coco.loadImgs(img_id)[0]['file_name'].replace(
            ".jpg", "-0.png")
        bbox_path = Path(self.box_location, bbox_name)
        bbox.save(bbox_path)

    def _filter_coordinate_boxes(self, img_id, ann):
        try:
            if self.img_id == img_id:
                self.coord_box = np.vstack(
                    [self.coord_box, np.asarray(ann["bbox"])])
            else:
                bbox = np.asarray(self.coord_box, dtype=np.uint8)
                bbox = Image.fromarray(bbox).convert("L")
                bbox_name = self.coco.loadImgs(
                    self.img_id)[0]['file_name'].replace(".jpg", "-0.png")
                bbox_path = Path(self.box_location, bbox_name)
                bbox.save(bbox_path)

                self.img_id = img_id
                self.coord_box = [0, 0, 0, 0]
                self.coord_box = np.vstack(
                    [self.coord_box, np.asarray(ann["bbox"])])
        except AttributeError:
            self.img_id = img_id
            self.coord_box = [0, 0, 0, 0]
            self.coord_box = np.asarray(ann["bbox"])

    def _fixed_ratio_bbox(self, seg_array, box_ratio):
        """ For example, let
        seg_array = [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        xb, yb, wb, hb = 0, 0, 5, 2 """
        _, _, wb, hb = self._get_seg_boundary(seg_array)
        rows, cols = np.nonzero(seg_array)
        ri = random.randint(0, len(rows) - 1)
        """ Then,
        y in [0, 1]
        x in [0, 1, 2, 3, 4] """
        y, x = rows[ri], cols[ri]
        """ We want
        h in [0, hb - y]
        w in [0, wb - x] """
        h, w = round((hb - y) * box_ratio), round((wb - x) * box_ratio)
        return x, y, w, h

    def _draw_bbox_from_seg(self, img_id, seg_array, box_ratio):
        img_height = self.coco.loadImgs(img_id)[0]['height']
        img_width = self.coco.loadImgs(img_id)[0]['width']
        seg = Image.fromarray(np.zeros((img_height, img_width)))
        draw = ImageDraw.Draw(seg)
        x, y, w, h = self._fixed_ratio_bbox(seg_array, box_ratio)
        """
        img = [
            [1, 1, 1],
            [0, 0, 0],
            [1, 1, 1],
        ]

        then (x, y) indexes of each element are given by [
            [(0, 0), (0, 1), (0, 2)],
            [(1, 0), (1, 1), (1, 2)],
            [(2, 0), (2, 1), (2, 2)],
        ]
        """
        rect = self._get_rect(x, y, w, h, 0)
        draw.polygon([tuple(p) for p in rect], fill=1)
        np_seg = np.asarray(seg, dtype=int)
        return np_seg

    def _filted_nonrandom_agg_boxes(self, img_id, ann):
        seg_array = self.coco.annToMask(ann)
        box_ratio = self.box_type["box ratio"]
        bbox = self._draw_bbox_from_seg(img_id, seg_array, box_ratio)
        for i in range(1, self.box_type["num miniboxes"]):
            bbox_ = bbox | self._draw_bbox_from_seg(img_id, seg_array,
                                                    box_ratio)
            bbox = bbox_

        bbox = np.array(bbox, dtype=np.uint8)
        bbox = Image.fromarray(bbox).convert("L")
        bbox_name = self.coco.loadImgs(img_id)[0]['file_name'].replace(
            ".jpg", "-0.png")
        bbox_path = Path(self.box_location, bbox_name)
        bbox.save(bbox_path)

    def _turn_mask_pixels_off(self, seg_array, ratio):
        rows, cols = np.where(seg_array == 1)
        rows = rows[:round(len(rows) * ratio)]
        cols = cols[:round(len(cols) * ratio)]
        seg_array[rows, cols] = 0
        return seg_array

    def _filter_turnedoff_mask(self, img_id, ann):
        seg_array = self.coco.annToMask(ann)
        ratio = self.box_type["ratio"]
        mask = self._turn_mask_pixels_off(seg_array, ratio)
        mask = np.array(mask, dtype=np.uint8)
        mask = Image.fromarray(mask).convert("L")

        width, height = mask.size
        width = round(width / sqrt(self.downsample))
        height = round(height / sqrt(self.downsample))
        mask = mask.resize((width, height), Image.ANTIALIAS)

        mask_name = self.coco.loadImgs(img_id)[0]['file_name'].replace(
            ".jpg", "-0.png")
        mask_path = Path(self.box_location, mask_name)
        mask.save(mask_path)

    def __del__(self):
        if self.box_type == "coordinate":
            bbox = np.asarray(self.coord_box, dtype=np.uint8)
            bbox = Image.fromarray(bbox).convert("L")
            bbox_name = self.coco.loadImgs(
                self.img_id)[0]['file_name'].replace(".jpg", "-0.png")
            bbox_path = Path(self.box_location, bbox_name)
            bbox.save(bbox_path)