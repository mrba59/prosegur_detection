# Copyright (c) Facebook, Inc. and its affiliates.
from pathlib import Path

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(config, opts, score_th):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(config)
    cfg.merge_from_list(opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = score_th
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_th
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = score_th
    cfg.freeze()
    return cfg


def init_model(config, opts, score_th):
    # initialize model
    cfg = setup_cfg(config, opts, score_th)
    model = DefaultPredictor(cfg)
    return model


def inference_image(model, image, path, score_th):
    # launch detections and read results
    all_predictions = []
    h, w, c = image.shape
    predictions = model(image)
    # read predictions from torch format annotations
    all_boxes = predictions['instances'].pred_boxes[:].tensor.tolist()
    all_classes = predictions['instances'].pred_classes[:].tolist()
    all_scores = predictions['instances'].scores[:].tolist()

    for bbox, classe, score in zip(all_boxes, all_classes, all_scores):
        #bbx_coco = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
        all_predictions.append([Path(path).name, h, w, bbox, score, classe])

    return all_predictions
