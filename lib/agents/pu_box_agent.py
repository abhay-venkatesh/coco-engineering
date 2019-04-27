from pathlib import Path


class PUBoxAgent:
    def run(self, config):
        raise NotImplementedError

    def _build(self, split, config):
        raise NotImplementedError

        if split not in ["train", "val"]:
            raise ValueError("Unsupported split. ")

        annotations_path = Path(config["source"], "annotations",
                                "stuff_" + split + "2017.json")
        coco = COCO(annotations_path)
