import os
from argparse import ArgumentParser
from ast import literal_eval
import json
import cv2
import pandas as pd
import logging
from datetime import datetime
import sys

date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")

FORMAT = '%(asctime)s %(message)s'
logging.basicConfig(format=FORMAT,
                    filename=f"../logs/log_draw/log_draw_{date}",
                    filemode='a',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
# add the handler to the root logger
logging.getLogger('').addHandler(console)

class_coco = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic_light',
              'fire_hydrant',
              'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
              'zebra', 'giraffe',
              'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports_ball',
              'kite', 'baseball_bat',
              'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork',
              'knife', 'spoon', 'bowl',
              'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut', 'cake',
              'chair', 'couch', 'potted_plant',
              'bed', 'dining_table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
              'oven', 'toaster',
              'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush']

class_prosegur = [None, 'cat', 'dog', 'person']


def draw_args():
    parser = ArgumentParser()
    parser.add_argument('annot_path',
                        help='path to annotation file coco (json) or opencv format (csv) with id class prosegur',
                        type=str)
    parser.add_argument('images_path', help='path to dir containing images', type=str)
    parser.add_argument('--output_dir', help='path to store output', type=str)
    parser.add_argument('--show', help='if True display image on screen', type=bool)
    parser.add_argument('--filename_list', nargs="+")
    parser.add_argument('--from_csv', help='read img from csv file', type=str)
    args = parser.parse_args()
    return args


# for draw_bbox_coco choose the original instances with coco category
def draw_bboxes(boxes, image, label, score):
    " function that draw bounding boxes onto the image given and add score, category"
    [x1, y1, w, h] = [int(boxes[0]), int(boxes[1]), int(boxes[2]), int(boxes[3])]
    c1 = (x1, y1)
    c2 = (x1 + w, y1 + h)
    cv2.rectangle(image, c1, c2, (255, 0, 0), 3)

    # if the text is out of image bounding , replace it
    if y1 < 30:
        yput_text = y1 + 15
    else:
        yput_text = y1 - 10
    if score:
        cv2.putText(image, label + '  ' + str(round(score, 2)), (x1, yput_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)
    else:
        cv2.putText(image, label, (x1, yput_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)


def check_image_in_dir(images_path, list_fn):
    # check images given by filename_list argument are in directory
    check = all(item in os.listdir(images_path) for item in list_fn)
    if not check:
        not_in_dir = [img for img in list_fn if img not in os.listdir(images_path)]
        logging.info(f"images {not_in_dir} are not in directory ")
    images_in_dir = [os.path.join(args.images_path, file) for file in list_fn if file in os.listdir(images_path)]
    return images_in_dir


def check_all_annotations(data, datatype, images_path):
    # check if all images in dir are the same as images in annot
    if datatype == 'json':
        images_coco = pd.DataFrame.from_dict(data['images'])
        filename = images_coco['file_name']
    if datatype == 'csv':
        filename = data['file_name']
    if not any(i in list(filename) for i in os.listdir(images_path)):
        logging.error("all images inside directory are different from images in annotations path")
        sys.exit()


def get_annotation(data, img_name, datatype, images_coco, annot_coco):
    # get all annotations for one image
    if datatype == 'json':
        row_img = images_coco[images_coco['file_name'] == img_name]
        if len(row_img.index) == 1:
            id_img = row_img.loc[row_img.index[0], 'id']
            annot = annot_coco[annot_coco['image_id'] == id_img]
        else:
            # if len >1 means that there is multiple id for the same filename (duplicate)
            # merge annotations of all id
            list_id = set(row_img['id'])
            annot = pd.DataFrame(
                columns=['segmentation', 'iscrowd', 'area', 'image_id', 'bbox', 'category_id', 'id'])
            for i in list_id:
                annot = pd.concat([annot, annot_coco[annot_coco['image_id'] == i]])

    if datatype == 'csv':
        annot = data[data['file_name'] == img_name]
    if len(annot) == 0:
        logging.info(f"image {img_name} has no annotations")
        value = False
    else:
        value = True
    return value, annot


def convert_coco_annot(annotations):
    # convert annotations opencv format to coco
    annot = pd.DataFrame(
        columns=['bbox', 'category_id', 'score'])
    i = 0
    for index, row in annotations.iterrows():
        if type(row['boxes']) == list:
            bboxes = row['boxes']
        else:
            # when reading data from csv file , list are considered as string , so we need to convert it to list
            bboxes = literal_eval(row['boxes'])

        bboxes = [bboxes[0], bboxes[1], bboxes[2] - bboxes[0], bboxes[3] - bboxes[1]]
        coco_category = row['coco_category']
        score = row['scores']
        annot.loc[i] = pd.Series({'bbox': bboxes, 'category_id': coco_category, "score": score})
        i += 1
    return annot


def read_annotations(annot_path):
    # read annotations file
    if 'csv' in annot_path:
        data = pd.read_csv(annot_path)
        data_type = 'csv'
        images_coco = None
        annot_coco = None
        categories = None
    # read annotations from json file
    elif 'json' in annot_path:
        data_type = 'json'
        with open(annot_path, ) as f:
            data = json.load(f)
        images_coco = pd.DataFrame.from_dict(data['images'])
        annot_coco = pd.DataFrame.from_dict(data['annotations'])
        categories = pd.DataFrame.from_dict(data['categories'])
    return data, data_type, images_coco, annot_coco, categories


def extract_draw(image_path, img_name, data_type, annot, categories):
    # extract boxes score and category from annotation and draw on image
    image = cv2.imread(image_path)
    if data_type == 'csv':
        annot = convert_coco_annot(annot)
    for index, raw in annot.iterrows():
        bboxes = raw['bbox']
        coco_category = raw['category_id']
        if data_type == 'csv':
            label = class_coco[coco_category]
            score = raw["score"]
        else:
            label = categories[categories['id'] == coco_category]['name'].values[0]
            score = None
        draw_bboxes(bboxes, image, label, score)
    if output_dir:
        output_path = os.path.join(output_dir, img_name)
        cv2.imwrite(output_path, image)
        # show image drawn
    if show:
        cv2.imshow('image', image)
        cv2.waitKey(0)


if __name__ == "__main__":
    args = draw_args()
    images_path = args.images_path
    annot_path = args.annot_path
    output_dir = args.output_dir
    list_fn = args.filename_list
    from_csv = args.from_csv
    show = args.show
    if output_dir and not os.path.exists(output_dir):
        print('creating dir')
        os.makedirs(output_dir)
    # get image path given in filename_list argument
    if list_fn:
        # get images path and check if images given in list are in directory
        images_in_dir = check_image_in_dir(images_path, list_fn)
    # get all images from directory
    elif from_csv:
        filelist_csv = pd.read_csv(from_csv)
        images_in_dir = check_image_in_dir(images_path, list(filelist_csv['file_name']))
    else:
        # get images path
        images_in_dir = [os.path.join(args.images_path, file) for file in os.listdir(args.images_path)]
    if len(images_in_dir) == 0:
        logging.error("none of images from list are in the directory ")
        sys.exit()
    # read load annotations
    data, data_type, images_coco, annot_coco, categories_coco = read_annotations(annot_path)
    # check if images from annotations are in the directory
    check_all_annotations(data, data_type, images_path)
    for file_path in images_in_dir:
        img_name = os.path.basename(file_path)
        value, annot = get_annotation(data, img_name, data_type, images_coco, annot_coco)
        if value:
            extract_draw(file_path, img_name, data_type, annot, categories_coco)
