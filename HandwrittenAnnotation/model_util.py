from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Input, Lambda
from tensorflow.python.keras.models import Model

from .model import yolo_body, yolo_loss


def create_model(input_shape, core_batch_size, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/yolo_weights.h5', num_channels=3):
    """create the training model"""
    K.clear_session()  # get a new session
    h, w = input_shape
    image_input = Input(shape=(h, w, num_channels))
    num_anchors = len(anchors)

    # y_true = [Input(shape=(h//{0: 32, 1: 16, 2: 8}[l], w//{0: 32, 1: 16, 2: 8}[l],
    #                       num_anchors//3, 5 + 1)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    # 転移学習
    # if load_pretrained:
    #    model_body.load_weights(weights_path, by_name=True)  # skip_mismatch=True
    #    print('Load weights {}.'.format(weights_path))
    #    if freeze_body in [1, 2]:
    #        # Freeze darknet53 body or freeze all but 3 output layers.
    #        num = (185, len(model_body.layers)-3)[freeze_body-1]
    #        for i in range(num): model_body.layers[i].trainable = False
    #        print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    # model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
    #                    arguments={'anchors': anchors, 'num_classes': num_classes, 'batch_size': core_batch_size,
    #                               'ignore_thresh': 0.5})([*model_body.output, *y_true])
    # model_loss = Lambda(lambda t: K.stack([t, K.constant(0)]))(model_loss)
    # model = Model([model_body.input, *y_true], model_loss)
    model = Model(model_body.input, model_body.output)

    return model


def get_loss_lambda(anchors, num_classes, core_batch_size):
    def loss(y_true, y_pred):
        yolo_loss(y_true, y_pred, anchors, num_classes, batch_size=core_batch_size, ignore_thresh=0.5)
    return loss
