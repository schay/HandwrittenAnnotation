# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
from timeit import default_timer as timer

import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Input
from .model import yolo_eval, yolo_body, tiny_yolo_body
from .utils import letterbox_image
from tensorflow.python.keras.utils import multi_gpu_model


class YOLO(object):
    _defaults = {
        # "model_path": 'model_data/yolo.h5',
        # "anchors_path": 'model_data/yolo_anchors.txt',
        # "classes_path": 'model_data/coco_classes.txt',

        "model_path": 'logs/000/ep003-loss42.700-val_loss48.123.h5',
        "anchors_path": 'model_data/tiny_yolo_anchors.txt',
        "classes_path": 'model_data/voc_classes.txt',

        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    def __init__(self, model_path, anchors_path, classes_names, model_image_size, gpu_num=0, num_channels=3):
        # self.class_names = self._get_class(classes_path)
        self.class_names = classes_names
        self.anchors = self._get_anchors(anchors_path)
        self.sess = K.get_session()
        self.gpu_num = gpu_num
        # self.score = 0.3
        self.score = 0.1
        self.iou = 0.45
        self.model_image_size = list(model_image_size)
        self.num_channels = num_channels
        self.boxes, self.scores, self.classes = self.generate(model_path)

    def _get_class(self, classes_path):
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self, anchors_path):
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self, model_path):
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        # self.yolo_model = load_model(model_path, compile=False)
        self.yolo_model = yolo_body(Input(shape=(None,None,self.num_channels)), num_anchors//3, num_classes)
        self.yolo_model.load_weights(model_path)

        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image, image_mode='RGB'):
        start = timer()
        '''
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)), image_mode=image_mode)
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size, image_mode=image_mode)

        image_data = np.array(boxed_image, dtype='float32')
        if image_mode == 'L':
            image_data = image_data[..., np.newaxis]
        else:
            image_data = image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        '''

        # サイズを合わせる
        boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)), image_mode=image_mode)
        image_data = np.array(boxed_image, dtype='float32')
        image_data = image_data[np.newaxis, ..., np.newaxis]
        image = image_data

        # feed_dict
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image,
                self.input_image_shape: [image.shape[1], image.shape[2]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        # font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
        #            size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        # thickness = (image.size[0] + image.size[1]) // 300

        r_out_boxes = []
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            # draw = ImageDraw.Draw(image)
            # label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.shape[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.shape[2], np.floor(right + 0.5).astype('int32'))
            # bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            # right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            r_out_boxes.append([left, top, right, bottom])

            '''
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw
            '''

        end = timer()
        print(end - start)
        return image, np.array(r_out_boxes), out_scores, out_classes

    def close_session(self):
        self.sess.close()
