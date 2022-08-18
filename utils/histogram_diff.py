import json
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def hist_parser():
    parser = ArgumentParser()
    parser.add_argument('path_annot_coco', help=' the annotation from open data in coco format ')
    parser.add_argument('path_predictions', help=' the annotation from prediction in coco format ')
    args = parser.parse_args()
    return args


def mapping_id_filename(instances_coco_orig, instances_coco_pred):
    # dict that contain as key, id of image, and as value filename from prediction instances
    image_id_to_filename_pred = {}
    # dict that contain, as keys filename and as value a list of image_id
    # from original instances in order to find duplicate
    filename_to_image_id_orig = {}

    # mapping from original data
    for image_info in instances_coco_orig['images']:
        if image_info['file_name'] not in list(filename_to_image_id_orig.keys()):
            filename_to_image_id_orig[image_info['file_name']] = [image_info['id']]
        else:
            # duplicate , same filename but different id
            filename_to_image_id_orig[image_info['file_name']].append(image_info['id'])

    # mapping from prediction data
    for image_info in instances_coco_pred['images']:
        image_id_to_filename_pred[image_info['id']] = image_info['file_name']

    return image_id_to_filename_pred, filename_to_image_id_orig


def get_annotations_count_per_category(image_id_to_filename_pred, filename_to_image_id_orig, instances_coco_orig,
                                       instances_coco_pred,
                                       class_prosegur):
    """
    function that will count for each filename the number of annotations per category for predicted and original results.
    return a dataframe with filename as index and count_dog_orig , count-dog_pred,  count_person_orig ... as column
     """
    #  Initialize dataframe with index equal list of original filename and all count columnn  to a list of zero
    df_count = pd.DataFrame()
    df_count['filename'] = filename_to_image_id_orig.keys()
    df_count['count_dog_orig'] = [0] * len(filename_to_image_id_orig.keys())
    df_count['count_cat_orig'] = [0] * len(filename_to_image_id_orig.keys())
    df_count['count_person_orig'] = [0] * len(filename_to_image_id_orig.keys())
    df_count['count_cat_pred'] = [0] * len(filename_to_image_id_orig.keys())
    df_count['count_dog_pred'] = [0] * len(filename_to_image_id_orig.keys())
    df_count['count_person_pred'] = [0] * len(filename_to_image_id_orig.keys())
    df_count = df_count.set_index('filename')

    # for each image,  iterate over original annotations corresponding to this image,
    # and get the number of annotations per category, and join the annotations duplicated
    for key, value in filename_to_image_id_orig.items():
        annot_im = [annot for annot in instances_coco_orig['annotations'] if annot['image_id'] in value]

        # divide the list of annotations per category (cat, dog, person)
        df_count.at[key, 'count_dog_orig'] = len(
            [annot for annot in annot_im if annot['category_id'] == class_prosegur['dog']])
        df_count.at[key, 'count_cat_orig'] = len(
            [annot for annot in annot_im if annot['category_id'] == class_prosegur['cat']])
        df_count.at[key, 'count_person_orig'] = len(
            [annot for annot in annot_im if annot['category_id'] == class_prosegur['person']])

    # for each image,  iterate over predicted annotations corresponding to this image,  and get the number of annotations per category
    annot_pred = instances_coco_pred['annotations']
    for annot in annot_pred:
        if annot['category_id'] == class_prosegur['dog']:
            df_count.at[image_id_to_filename_pred[annot['image_id']], "count_dog_pred"] += 1
        elif annot['category_id'] == class_prosegur['cat']:
            df_count.at[image_id_to_filename_pred[annot['image_id']], "count_cat_pred"] += 1
        elif annot['category_id'] == class_prosegur['person']:
            df_count.at[image_id_to_filename_pred[annot['image_id']], "count_person_pred"] += 1

    return df_count


def get_count_diff_per_category(df):
    """
    for each image , calcul the difference between the number of annotations per category between original and predicted
    """

    values_dog = (df['count_dog_pred'] - df['count_dog_orig']).loc[
        ~((df['count_dog_orig'] == 0) & (df['count_dog_pred'] == 0))]
    values_cat = (df['count_cat_pred'] - df['count_cat_orig']).loc[
        ~((df['count_cat_orig'] == 0) & (df['count_cat_pred'] == 0))]
    values_person = (df['count_person_pred'] - df['count_person_orig']).loc[
        ~((df['count_person_orig'] == 0) & (df['count_person_pred'] == 0))]

    return values_person, values_cat, values_dog


def pred_none_orig_1(df):
    """
    functions that extract images that have 0 annotations in predicted results and at least one in original results
    """
    # conditions at least 1 annotations for original and 0 for predicted
    df_dog_pred_none = df.loc[
        (df['count_dog_orig'] > 0) & (df['count_dog_pred'] == 0), ['count_dog_orig', 'count_dog_pred']]

    df_cat_pred_none = df.loc[
        (df['count_cat_orig'] > 0) & (df['count_cat_pred'] == 0), ['count_cat_orig', 'count_cat_pred']]

    df_person_pred_none = df.loc[
        (df['count_person_orig'] > 0) & (df['count_person_pred'] == 0), ['count_person_orig', 'count_person_pred']]

    # save results in csv
    df_dog_pred_none.to_csv('../pred_none_orig_1or_more/dog_pred_none.csv')
    df_cat_pred_none.to_csv('../pred_none_orig_1or_more/cat_pred_none.csv')
    df_person_pred_none.to_csv('../pred_none_orig_1or_more/person_pred_none.csv')

    return df_dog_pred_none, df_cat_pred_none, df_person_pred_none


if __name__ == "__main__":
    args = hist_parser()
    with open('mapping.json', ) as j1:
        class_prosegur = json.load(j1)
    # path to the original annotations file from open_image in coco format
    path_instances_original = args.path_annot_coco
    # path to the annotations file generated by our model in coco format
    path_instances_model = args.path_predictions

    with open(path_instances_original, ) as f1:
        instances_coco_orig = json.load(f1)

    with open(path_instances_model, ) as f2:
        instances_coco_pred = json.load(f2)

    # get mapping of filename to image_id for original instances , and image_id to filename for predictions instances
    image_id_to_filename_pred, filename_to_image_id_orig = mapping_id_filename(instances_coco_orig, instances_coco_pred)
    # get a dataframe that count number of annotations per category , for original and prediction instances
    df_count = get_annotations_count_per_category(image_id_to_filename_pred,
                                                  filename_to_image_id_orig,
                                                  instances_coco_orig, instances_coco_pred,
                                                  class_prosegur)
    # get the difference of annotations number between original end predicted for each category
    diff_person, diff_cat, diff_dog = get_count_diff_per_category(df_count)
    # extract images that have 0 annotations in predicted results and at least one in original results
    df_dog_pred_none, df_cat_pred_none, df_person_pred_none = pred_none_orig_1(df_count)
    # plot and save figure that represents the repartition of annotations number difference
    # add title and x label
    fig_person = sns.histplot(data=diff_person, x=diff_person.values)
    fig_person.set(xlabel ="diff", title ='Distribution of the difference (person)')
    plt.savefig('../plot_diff/person_count_diff.png')
    plt.clf()
    fig_dog = sns.histplot(data=diff_dog, x=diff_dog.values)
    fig_dog.set(xlabel ="diff", title ='Distribution of the difference (dog)')
    plt.savefig('../plot_diff/dog_count_diff.png')
    plt.clf()
    fig_cat = sns.histplot(data=diff_cat, x=diff_cat.values)
    fig_cat.set(xlabel ="diff", title ='Distribution of the difference (cat)')
    plt.savefig('../plot_diff/cat_count_diff.png')
    plt.clf()
