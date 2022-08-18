import logging
from pathlib import Path

import torch
from mmdet.apis import inference_detector, init_detector


def read_results_mmcv(results, score_th, img, h, w, img_id):
    """ convert results stored as mmcv format to list """
    filename = Path(img)
    annot = []
    for i, value in enumerate(results):
        for bbox_score in value:
            bbx = [point for point in bbox_score[:4]]
            #bbx_coco = [bbx[0], bbx[1], bbx[2]-bbx[0], bbx[3]-bbx[1]]
            score = bbox_score[-1]
            if score > score_th:
                data_csv = [filename.name, h, w, bbx, score, i]
                annot.append(data_csv)
    return annot


def init_model(config, checkpoint):
    # initialize model
    model = init_detector(config, checkpoint, device=torch.device('cuda:0'))
    logging.info('load config ' + str(config) + ' and weights ' + str(checkpoint))
    return model


def inference_image(model, image, path, score_th):
    # launch detection and get results as list
    h, w, c = image.shape
    results = inference_detector(model, image)
    annot_csv = read_results_mmcv(results[0], score_th, path, h, w, None)
    return annot_csv
