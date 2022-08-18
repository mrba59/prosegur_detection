import os
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


def filter_parse():
    parser = ArgumentParser()
    parser.add_argument('--score_th', help=' score thresold')
    parser.add_argument('--class_filter', help=' list of coco label we want to keep  ', nargs="+")
    parser.add_argument('--csv_path', help=' path to to csv file that contains detection results  ')
    args = parser.parse_args()
    return args


def filter_class_score(prediction, output_dir, score_th, class_filter):
    """ filter the results of detection , by class and score_threshold """
    #  drop line by score threshold
    for index, row in prediction.iterrows():
        if float(row['scores']) < float(score_th):
            prediction = prediction.drop(index)
    # drop line by class
    if class_filter:
        for index, row in prediction.iterrows():
            if row["label_coco"] not in class_filter:
                prediction = prediction.drop(index)
    # define name of csv_file
    if class_filter:
        write_csv = ""
        for cl in class_filter:
            write_csv = write_csv + "_" + str(cl)
    else:
        write_csv = 'all_class'

    prediction.to_csv(os.path.join(output_dir, "filter_" + str(score_th) + '_' + write_csv + '.csv'), index=False)
    return prediction


if __name__ == "__main__":
    args = filter_parse()

    score_th = args.score_th
    csv_path = args.csv_path
    csv_path = Path(csv_path)
    fname = csv_path.stem
    output_dir = csv_path.parent
    data = pd.read_csv(csv_path)
    class_filter = args.class_filter

    filter_class_score(data, output_dir, score_th, class_filter)
