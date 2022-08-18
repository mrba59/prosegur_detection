import json
from argparse import ArgumentParser
from ast import literal_eval
import pandas as pd


def convert_parse():
    parser = ArgumentParser()
    parser.add_argument('csv_path', help=' path to csv file containing detections ')
    parser.add_argument('json_path', help=' path to store annotations at format coco  ')
    args = parser.parse_args()
    return args


def images_annot_coco(row):
    # build the coco key images
    image = {}
    image["height"] = row.h
    image["width"] = row.w
    image["id"] = row.id_image
    image["file_name"] = row.file_name
    image["license"] = 1
    image["flickr_url"] = ""
    image["coco_url"] = ""
    image["date_captured"] = ""
    return image


def category_annot_coco(row):
    # build the coco key category
    category = {}
    category["supercategory"] = 'None'
    category["id"] = row.categoryid
    category["name"] = row[7]
    return category


def annotations_coco(row):
    # build the coco keys annotations
    annotation = {}
    # when reading from csv , if a column contain lists, they are considered as string,
    # we need to convert it back to list
    if type(row.boxes) != list:
        bbox = literal_eval(row.boxes)
    else:
        bbox = row.boxes
    area = (float(bbox[2]) - float(bbox[0])) * (float(bbox[3]) - float(bbox[1]))
    bbox=[float(bbox[0]), float(bbox[1]), float(bbox[2]) - float(bbox[0]), float(bbox[3]) - float(bbox[1])]
    annotation["segmentation"] = []
    annotation["iscrowd"] = 0
    annotation["area"] = area
    annotation["image_id"] = row.id_image

    annotation["bbox"] = bbox

    annotation["category_id"] = row.coco_category
    annotation["id"] = row.annid
    return annotation


def convert(data, output):
    """
    convert csv file that contain detection into coco format
    """

    # coco keys
    images = []
    categories = []
    annotations = []

    info = {"year": 2022, "url": "https://www.prosegur.es/", "contributor": "Prosegur",
            "date_created": "2022-03-22 15:03:27"}
    licenses = [{"id": 1, "name": "Creative Commons Attribution 4.0 International",
                 "url": "https://creativecommons.org/licenses/by/4.0/"},
                {"id": 2, "name": "Creative Commons Attribution 2.0 Generic",
                 "url": "https://creativecommons.org/licenses/by/2.0/"}]

    # initialize category
    categories = [{
        "id": 1,
        "name": "cat",
        "supercategory": "object"
    }, {
        "id": 2,
        "name": "dog",
        "supercategory": "object"
    }, {
        "id": 3,
        "name": "person",
        "supercategory": "object"
    }]

    # get filename and category id
    data['fileid'] = data['file_name'].astype('category').cat.codes
    data['categoryid'] = data['coco_category']
    # get id annotations
    data['annid'] = data.index

    for row in data.itertuples():
        annotations.append(annotations_coco(row))

    imagedf = data.drop_duplicates(subset=['fileid']).sort_values(by='fileid')
    for row in imagedf.itertuples():
        images.append(images_annot_coco(row))
    # build th json file that will contain annotations to coco format
    data_coco = {}
    data_coco["info"] = info
    data_coco["images"] = images
    data_coco["categories"] = categories
    data_coco["annotations"] = annotations
    data_coco["licenses"] = licenses
    json.dump(data_coco, open(output, "w"), indent=4)


if __name__ == "__main__":
    args = convert_parse()
    path = args.csv_path
    save_json_path = args.json_path
    data = pd.read_csv(path)
    convert(data, save_json_path)
