import argparse
import logging
import os
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import cv2
import pandas as pd

from utils import MultipleDetector

date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")

FORMAT = '%(asctime)s '
logging.basicConfig(format='%(asctime)s %(message)s',
                    filename=f"logs/log_{date}",
                    filemode='a',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger('multiple detector')


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img_dir', help=' image path for signle inference or path to dir for multiple  ')
    parser.add_argument('config', help='Config file')
    parser.add_argument('output_dir', help='result output dir ')
    parser.add_argument('model', type=str, help='choose model to run between [pedestron , detectron, swin ] ')
    parser.add_argument('score_th', type=float, help='sccore threshold ', default=0.1)
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument('--video', help=' path of the video or 0 for camera ')
    parser.add_argument('--draw', help=' if true show image', type=bool)
    parser.add_argument('--output_video', help='if true show detetection by recording video', type=bool )
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument("--opts",
                        help="checkpoint file for detectron2 Modify config options using the command-line 'KEY VALUE' pairs",
                        default=[],
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args


def run_detection(input_f, output_dir, model_name, checkpoint, config, score_th, draw, video, output_video):
    """ run detections, the input is either a video  or a directory of image or filepath """
    # list that will contain all predictions as [ 'file_name', 'h', 'w', 'boxes', 'scores', 'coco_category', 'id_image']
    all_predictions = []
    if video:
        # reading from video
        logger.info('reading data from video')
        path_vid = video.split(".")[0]
        vid = cv2.VideoCapture(args.video)
        if vid.isOpened() == False:
            logger.error("Error opening the video file")
        count = 0
        frame_width, frame_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vid.get(cv2.CAP_PROP_FPS)
        writer = None
        # store the results of detections into a video
        if output_video:

            writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                     (frame_width, frame_height))
        start = time.time()
        # initialize the model
        model = MultipleDetector.MultipleDetector(input_f, output_dir, model_name, checkpoint, config, score_th)
        while True:
            # load image
            ret, image = model.read_images(vid)
            if ret:
                # launch detections
                results_csv = model.inference(image, path_vid + '/frame_' + str(count))
                if results_csv:
                    for line in results_csv:
                        all_predictions.append(line)
                count = count + 1
                # draw bboxes on image
                img_draw = model.draw_bb(results_csv, image, 0.3)
                # debug mode to show detections
                if draw or path_vid == 'webcam':
                    cv2.imshow('draw', img_draw)
                    if cv2.waitKey(1) == ord('q'):
                        break
                if writer:
                    # write image into output_video path
                    writer.write(img_draw)
            else:
                break

        end = time.time()
        vid.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    else:
        if os.path.isfile(input_f):
            # reading single image
            logger.info('reading single image')
            path_images = [input_f]
        elif os.path.isdir(input_f):
            # reading images from directory
            logger.info('reading images from directory')
            # get list of images path
            path_images = [os.path.join(input_f, file) for file in os.listdir(input_f)]
        else:
            logger.error('error argument input_f must be path to image or path to directory' )

        # initiate model
        start = time.time()
        model = MultipleDetector.MultipleDetector(input_f, output_dir, model_name, checkpoint, config, score_th)
        for img_id, path in enumerate(path_images):
            # load image
            fn = os.path.basename(path)
            image = model.read_images(path)
            # launch detections
            results_csv = model.inference(image, path)
            if results_csv:
                for line in results_csv:
                    # add image_id to detection
                    line.append(img_id)
                    all_predictions.append(line)
            # debug mode to show detections
            if draw:
                img_draw = model.draw_bb(results_csv, image, 0.3)
                output_path = os.path.join(output_dir, model_name + '/images/' + fn)
                cv2.imwrite(output_path, img_draw)
                cv2.imshow('draw', img_draw)
                if cv2.waitKey(500) == ord('q'):
                    break
        end = time.time()

    # store results to a dataframe
    if len(all_predictions[0]) == 7:
        df_results = pd.DataFrame(all_predictions,
                                  columns=['file_name', 'h', 'w', 'boxes', 'scores', 'coco_category', 'id_image'])
    elif len(all_predictions[0]) == 6:
        df_results = pd.DataFrame(all_predictions, columns=['file_name', 'h', 'w', 'boxes', 'scores', 'coco_category'])

    # add label column to the dataframe ( dog cat person , cellphone ...)
    label_column = []
    for index, row in df_results.iterrows():
        label_column.append(model.coco_classname[row['coco_category']])
    df_results['label_coco'] = label_column

    if output_video:
        path_res = os.path.join(output_dir, str(model_name) + '/videos/results.csv')
    else:
        path_res = os.path.join(output_dir, str(model_name) + '/annotations/annotations_csv/results.csv')
    df_results.to_csv(path_res, index=False)

    if_time = end - start
    logger.info('inference time :' + str(if_time))
    logger.info('saved results at ' + path_res)
    return df_results


if __name__ == "__main__":
    args = parse_args()
    if args.checkpoint:
        checkpoint = args.checkpoint
    elif args.opts:
        checkpoint = args.opts
    else:
        logger.error('error no checkpoint given')
    if args.output_video:
        fn = Path(args.video).stem
        output_video = os.path.join(args.output_dir, args.model+'/videos/' + fn + '_out.mp4')
    else:
        output_video = None
    run_detection(args.img_dir, args.output_dir, args.model, checkpoint, args.config, args.score_th,
                  args.draw, args.video, output_video)
