import os
from pathlib import Path
import logging
import pandas as pd
from argparse import ArgumentParser
from datetime import datetime
from utils import filter_class_score
from utils import mapping
from utils import convert_to_coco

date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")

FORMAT = '%(asctime)s '
logging.basicConfig(format='%(asctime)s %(message)s',
                    filename=f"logs/log_{date}",
                    filemode='a',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger('multiple detector')


def pipeline_args():
    parser = ArgumentParser()
    parser.add_argument('score_filter', type=float)
    parser.add_argument('path_to_csv', help='path to csv that contain results ')
    parser.add_argument('--class_filter', nargs="+")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = pipeline_args()
    score_filter = args.score_filter
    class_list = args.class_filter
    csv_path = Path(args.path_to_csv)
    df_results = pd.read_csv(csv_path)
    output_dir = csv_path.parent

    # launch the filtering by score and class
    df_res_filtered = filter_class_score.filter_class_score(df_results, output_dir, score_filter, class_list)

    # launch the mapping to prosegur classes
    df_filtered_mapped = mapping.mapping(df_res_filtered, output_dir)

    # launch the conversion from csv to coco format json
    output_dir_coco = str(output_dir).replace('annotations_csv', '')
    output_annot = os.path.join(output_dir_coco, 'annotations_coco/instances_' + str(score_filter) + '.json')
    convert_to_coco.convert(df_filtered_mapped, output_annot)
