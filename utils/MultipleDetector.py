import sys

import cv2


class MultipleDetector:
    """ Class that init and return model  , either pedestron , detectron2 or swin,  and launch inference on given input     """

    def __init__(self, input_f, output_dir, model_name, checkpoint, config, score_th):

        self.input_f = input_f  # images input,  path to file , or dir
        self.model_name = model_name  # model name [ pedestron detectron2 , swin]
        self.config = config  # configuration of model
        self.checkpoint = checkpoint  # weights of pretrained models
        self.output_dir = output_dir  # dir to store the output results
        self.score_th = score_th  # score threshold
        self.coco_classname = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                               'traffic_light', 'fire_hydrant',
                               'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                               'elephant', 'bear', 'zebra', 'giraffe',
                               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
                               'sports_ball', 'kite', 'baseball_bat',
                               'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket', 'bottle', 'wine_glass',
                               'cup', 'fork', 'knife', 'spoon', 'bowl',
                               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot_dog', 'pizza',
                               'donut', 'cake', 'chair', 'couch', 'potted_plant',
                               'bed', 'dining_table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                               'cell_phone', 'microwave', 'oven', 'toaster',
                               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy_bear', 'hair_drier',
                               'toothbrush']

        # initialize model
        if self.model_name == 'pedestron':
            sys.path.insert(0, 'models/Pedestron')
            from utils import pedestron
            self.model = pedestron.init_model(self.config, self.checkpoint)

        elif self.model_name == 'detectron2':
            sys.path.insert(0, 'models/detectron2/demo')
            from utils import detectron2
            self.model = detectron2.init_model(self.config, self.checkpoint, self.score_th)

        elif self.model_name == 'swin':
            sys.path.insert(0, 'models/Swin')
            from utils import swin
            self.model = swin.init_model(self.config, self.checkpoint)

    def read_images(self, data):
        # read image from path , or frame from video
        if isinstance(data, str):
            image = cv2.imread(data)
            return image
        else:
            r, frame = data.read()
            return r, frame

    def inference(self, image, input_f):

        """inference image
        input_f can be path to file , list of file or dict
        return dataframe with results of prediction """

        if self.model_name == 'pedestron':

            from utils import pedestron
            csv_file = pedestron.inference_image(self.model, image, input_f, self.score_th)

        elif self.model_name == 'detectron2':

            from utils import detectron2
            csv_file = detectron2.inference_image(self.model, image, input_f, self.score_th)

        elif self.model_name == 'swin':

            from utils import swin
            csv_file = swin.inference_image(self.model, image, input_f, self.score_th)
        else:
            print('error model name not recognized , choose between [detectron2, swin, pedestron)')

        return csv_file

    def draw_bb(self, list_bbox, image, score):
        # drax bboxes on image and filter by score
        for pred in list_bbox:
            if float(pred[4]) > float(score):
                # get 2 corner of bounding boxes
                [x1, y1, w, h] = pred[3]
                c1 = (int(x1), int(y1))
                c2 = (int(x1)+int(w), int(y1)+int(h))
                height, width = pred[1], pred[2]
                cv2.rectangle(image, c1, c2, (255, 0, 0), 3)
                if y1 < 30:
                    yput_text = int(y1 + 15)
                else:
                    yput_text = int(y1 - 10)
                class_name = str(self.coco_classname[int(pred[5])])
                sc = str(round(float(pred[4]), 2))
                cv2.putText(image, class_name + '  ' + sc, (int(x1), int(yput_text)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (36, 255, 12), 2)

        return image
