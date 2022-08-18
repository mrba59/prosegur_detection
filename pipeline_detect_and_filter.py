import argparse
import os
import logging

from argparse import ArgumentParser
from datetime import datetime

import detection
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

parser = ArgumentParser()
parser.add_argument('img', help=' image path for signle inference or path to dir for multiple  ')
parser.add_argument('config', help='Config file')
parser.add_argument('output_dir', help='result output dir ')
parser.add_argument('model', type=str, help='choose model to run between [pedestron , detectron, swin ] ')
parser.add_argument('score_th', type=float, help='sccore threshold ', default=0.1)
parser.add_argument('score_filter', type=float)
parser.add_argument('--video', help=' path of the video or 0 for camera ')
parser.add_argument('--checkpoint', help='Checkpoint file')
parser.add_argument('--output_video', type=str, help='if reading from video , show detection in video')
parser.add_argument('--draw', type=bool, help='if true show detections ')
parser.add_argument('--device', default='cuda:0', type=str, help='Device used for inference')
parser.add_argument('--class_filter', nargs="+")
parser.add_argument('--opts',
                    help="checkpoint file for detectron2 Modify config options using the command-line 'KEY VALUE' pairs",
                    default=[],
                    nargs=argparse.REMAINDER)
args = parser.parse_args()

if __name__ == "__main__":

    input_f = args.img
    output_dir = args.output_dir
    model_name = args.model
    config = args.config
    score_th = args.score_th
    score_filter = args.score_filter
    class_list = args.class_filter
    draw = args.draw
    video = args.video

    if args.output_video:
        if '.mp4' in args.video:
            output_video = os.path.join(args.output_video, args.model + '/videos/' + str(args.video))
        else:
            output_video = os.path.join(args.output_video, args.model + '/videos/' + str(args.video) + '.mp4')
    else:
        output_video = None

    if args.checkpoint:
        checkpoint = args.checkpoint
    elif args.opts:
        checkpoint = args.opts
    else:
        logger.error('error no checkpoint given')

    # launch the detections
    df_results = detection.run_detection(input_f, output_dir, model_name, checkpoint, config, 0.2, draw,
                                         video, output_video)
    # launch the filtering by score and class
    output_filter = os.path.join(output_dir, model_name + '/annotations/annotations_csv/')
    df_res_filtered = filter_class_score.filter_class_score(df_results, output_filter, score_filter, class_list)
    # launch the mapping to prosegur classes
    output_mapped = os.path.join(output_dir, model_name + '/annotations/annotations_csv/')
    df_filtered_mapped = mapping.mapping(df_res_filtered, output_mapped)
    # launch the conversion from csv to coco format json
    output_annot = str(output_mapped).replace('annotations_csv', '')
    output_annot = os.path.join(output_annot, 'annotations_coco/instances_' + str(score_filter) + '.json')
    convert_to_coco.convert(df_filtered_mapped, output_annot)
