import os.path
import json
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path


def mapping_parse():
    parser = ArgumentParser()
    parser.add_argument('csv_path', help=' path od csv file containing the predictions  ')
    args = parser.parse_args()
    return args


def mapping(predictions, output_dir):
    # load prosegur classes id
    with open('utils/mapping.json', ) as f:
        class_prosegur = json.load(f)

    # mapping between prosegur-class-id and coco id
    for key, value in class_prosegur.items():
        for index, row in predictions.iterrows():
            if key == row['label_coco']:
                predictions.loc[index, "coco_category"] = value

    predictions.to_csv(os.path.join(output_dir, 'mapped.csv'), index=False)
    return predictions


if __name__ == "__main__":
    args = mapping_parse()
    csv_path = args.csv_path
    csv_path = Path(csv_path)
    fname = csv_path.stem
    output_dir = csv_path.parent
    data = pd.read_csv(csv_path)
    mapping(data, output_dir)
