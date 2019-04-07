import numpy as np
from .model import preprocess_true_boxes
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties

from PIL import Image
from tensorflow.python.keras.utils import Sequence


def image_resize_rescale(image, background_size, translate, scale, image_mode='L'):
    if image_mode == 'L':
        image = np.uint8(image)
    new_size = image.shape * scale
    diff = (image.shape - new_size)/2
    pil_image = Image.fromarray(image)
    pil_image = pil_image.resize(
        [int(new_size[1]), int(new_size[0])], Image.BICUBIC)
    new_pil_image = Image.new(
        image_mode, [int(background_size[1]), int(background_size[0])])
    new_pil_image.paste(
        pil_image, (int(translate[1] + diff[1]), int(translate[0] + diff[0])))
    return np.asarray(new_pil_image)


def box_resize_rescale(box, background_size, translate, scale):
    w = ((box[2] - box[0]) * scale[1])
    h = ((box[3] - box[1]) * scale[0])
    c_x = (box[0] + box[2])/2 + translate[1]
    c_y = (box[1] + box[3])/2 + translate[0]
    new_box = np.array([c_x - w/2, c_y - h/2, c_x + w/2, c_y + h/2])
    new_box[0] = np.where(new_box[0] < 0, 0, new_box[0])
    new_box[1] = np.where(new_box[1] < 0, 0, new_box[1])
    new_box[2] = np.where(new_box[2] > background_size[1], background_size[1], new_box[2])
    new_box[3] = np.where(new_box[3] > background_size[0], background_size[0], new_box[3])
    return new_box


def random_augment(index, images, boxes, image_size, background_size, r_translate=0.2, r_scale=0.2, r_aspect_rate=0.2, image_type=np.uint8):
    n_image = len(index)
    translates = np.random.rand(n_image, 2)*r_translate*2 - r_translate
    scales = np.random.rand(n_image)*r_scale*2 - r_scale + 1
    aspect_rates = np.random.rand(n_image)*r_aspect_rate*2 - r_aspect_rate + 1
    positions = np.stack([np.ones([n_image])*(background_size[0] - image_size[0])/2, np.arange(
        n_image)*(background_size[1]/n_image)], axis=1)

    concat_new_image = np.zeros(background_size, image_type)
    new_boxes = []
    for c_index, i in zip(index, range(len(index))):
        scale = scales[i]
        aspect_rate = aspect_rates[i]
        translate = translates[i]
        position = positions[i]
        image, box = images[c_index], boxes[c_index]
        image = np.squeeze(image)

        t_scale = np.asarray([scale*aspect_rate, scale])
        t_translate = translate+position

        new_image = image_resize_rescale(image, background_size, t_translate, t_scale)
        new_box = box_resize_rescale(box, background_size, t_translate, t_scale)

        concat_new_image = np.logical_or(concat_new_image, new_image)
        new_boxes.append(new_box)

    return concat_new_image, np.array(new_boxes)


class RandomAugmentSequence(Sequence):

    def __init__(self, index, images, boxes, labels, batch_size, image_size, input_shape, anchors, num_classes, n_concat, r_translate=0.2, r_scale=0.5, r_aspect_rate=0.2):
        self.images = images
        self.boxes = boxes
        self.labels = labels
        self.index = shuffle(index)
        self.batch_size = batch_size
        self.image_size = image_size
        self.input_shape = input_shape
        self.anchors = anchors
        self.num_classes = num_classes
        self.n_concat = n_concat
        self.r_translate = r_translate
        self.r_scale = r_scale
        self.r_aspect_rate = r_aspect_rate

    def __len__(self):
        return int(np.ceil(len(self.index) / float(self.batch_size) / float(self.n_concat))) - 1

    def __getitem__(self, idx):
        batch_index = self.index[idx * self.batch_size * self.n_concat:(idx + 1) * self.batch_size * self.n_concat]

        image_data = []
        box_data = []
        for b in range(self.batch_size):
            b_index = batch_index[b * self.n_concat: (b + 1) * self.n_concat]

            image, box = random_augment(
                b_index, self.images, self.boxes, self.image_size, self.input_shape,
                r_translate=self.r_translate, r_scale=self.r_scale, r_aspect_rate=self.r_aspect_rate, image_type=np.bool)
            box_t = box.astype(np.uint16)
            if len(box_t.shape) < 2:
                print(idx)
                print(b_index)
                print(box.shape)
                print(box_t.shape)
            box_label = np.concatenate([box_t, self.labels[b_index][np.newaxis].T], axis=1)
            image_data.append(image)
            box_data.append(box_label)
        image_data = np.array(image_data)
        box_data = np.array(box_data)

        y_true = preprocess_true_boxes(
            box_data, self.input_shape, self.anchors, self.num_classes)

        # return [image_data[..., np.newaxis], *y_true], np.zeros(self.batch_size)
        return image_data[..., np.newaxis], y_true

    def on_epoch_end(self):
        self.index = shuffle(self.index)


def center_to_corner_box(b):
    return np.array([b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2])


def corner_to_center_box(b):
    return np.array([b[0] + (b[2] - b[0]) / 2, b[1] + (b[3] - b[1]) / 2, b[2] - b[0], b[3] - b[1]])


def get_anchors(anchors_path):
    """loads the anchors from a file"""
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def extraction_boxes(labels, threshould=0.5):
    boxes = labels[..., 0: 4]
    scores = labels[..., 4]
    classes = labels[..., 5:3035]

    idx = np.where(scores > threshould)
    return boxes[idx], scores[idx], classes[idx]


def show_image_labels(image, boxes, scores, classes, threshould=0.5):
    fp = FontProperties(fname='font/ipam.ttf');
    
    ax = plt.axes()
    plt.gray()
    plt.imshow(image)
    for box, score, class_ in zip(boxes, scores, classes):
        if score > threshould:
            color = 'r'
            width = box[2]*image.shape[1]
            height = box[3]*image.shape[0]
            point = [box[0]*image.shape[1] - width/2, box[1]*image.shape[0] - height/2]
            patch = patches.Rectangle(point, width, height, fill=False, color=color)
            ax.add_patch(patch)
            plt.text(point[0], point[1], class_, color=color, fontproperties = fp)
    plt.show()
