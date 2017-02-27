from __future__ import print_function

import sys,os,glob
sys.path.append('/home/jakub.czakon/deep_learning_art_webinar')

import time

import numpy as np

from matplotlib import pylab as plt
from PIL import Image

from scipy.optimize import fmin_l_bfgs_b

from keras.applications.vgg16 import VGG16
from keras.applications.imagenet_utils import preprocess_input,decode_predictions
from keras.preprocessing import image as img_utils
from keras.layers import Input
import keras.backend as K

import utils

from deepsense import neptune

# NEPTUNE CONFIG
def set_style_weight(value):
    global STYLE_WEIGHT
    STYLE_WEIGHT  = float(value)
    return str(STYLE_WEIGHT)
    
def set_content_weight(value):
    global CONTENT_WEIGHT
    CONTENT_WEIGHT  = float(value)
    return str(CONTENT_WEIGHT)
           
def neptune_image(raw_image,description):
    stylish_image = Image.fromarray(raw_image)
    return neptune.Image(
        name="neptune neural art",
        description=description,
        data=stylish_image)

ctx = neptune.Context()

logging_channel = ctx.job.create_channel(
    name='logging_channel',
    channel_type=neptune.ChannelType.TEXT)

loss_channel = ctx.job.create_channel(
    name='training loss',
    channel_type=neptune.ChannelType.NUMERIC)

base_channel = ctx.job.create_channel(
    name='base image',
    channel_type=neptune.ChannelType.IMAGE)

style_channel = ctx.job.create_channel(
    name='style image',
    channel_type=neptune.ChannelType.IMAGE)

combined_channel = ctx.job.create_channel(
    name='combined image',
    channel_type=neptune.ChannelType.IMAGE)

ctx.job.register_action(name='style', handler = set_style_weight)
ctx.job.register_action(name='content', handler = set_content_weight)


BASE_IMAGE_PATH = (ctx.params.base_file)
STYLE_IMAGE_PATH = (ctx.params.style_file)
STYLE_ITER = (ctx.params.style_iter)
OPTIM_ITER = (ctx.params.optim_iter)
CONTENT_WEIGHT = (ctx.params.content_weight)
STYLE_WEIGHT = (ctx.params.style_weight)

content_feature_names = 'block4_conv2'
style_feature_names = ['block1_conv1', 'block2_conv1','block3_conv1','block4_conv1',
                       'block5_conv1']

img_nrows,img_ncols = 224,224
img_size = (img_nrows, img_ncols, 3)

base_image = img_utils.load_img(BASE_IMAGE_PATH,target_size= img_size[:2])
base_image_tensor = utils.img2vggtensor(base_image)

style_image = img_utils.load_img(STYLE_IMAGE_PATH,target_size= img_size[:2])
style_image_tensor = utils.img2vggtensor(style_image)

base = K.variable(base_image_tensor)
style = K.variable(style_image_tensor)
combination= K.placeholder((1,)+img_size)

input_tensor = K.concatenate([base,style,combination], axis=0)

img_recognition_network = VGG16(input_tensor=input_tensor, 
                                weights='imagenet', 
                                include_top=True)

layer_dict = utils.get_layer_dict_name(img_recognition_network)

def gram_matrix(x):
    assert K.ndim(x) == 3
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

def content_loss(base, combination):
    return K.sum(K.square(combination - base))
    
def eval_loss_and_grads(x,layer_dict,content_weight,style_weight):
    x = x.reshape((1,) + img_size)
    
    content_features = layer_dict[content_feature_names].output

    base_image_features = content_features[0, :, :, :]
    combination_features = content_features[2, :, :, :]

    loss = K.variable(0.)

    # CONTENT LOSS
    loss += content_weight * content_loss(base_image_features,
                                          combination_features)

    # STYLE LOSS
    for layer_name in style_feature_names:
        layer_features = layer_dict[layer_name].output

        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]

        style_loss_chunk = style_loss(style_reference_features, combination_features)
        loss += (style_weight / len(style_feature_names)) * style_loss_chunk

    # get the gradients of the generated image wrt the loss
    grads = K.gradients(loss, combination)

    outputs = [loss] + grads

    grad_loss_func = K.function([combination], outputs)
    
    outs = grad_loss_func([x])
    
    loss_value = outs[0]
    grad_values = outs[1].flatten().astype('float64')

    return loss_value, grad_values


class Evaluator(object):
    def __init__(self,layer_dict,content_weight,style_weight):
        self.loss_value = None
        self.grad_values = None
        self.layer_dict = layer_dict
        self.content_weight = content_weight
        self.style_weight = style_weight
        
    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x,
                                                      layer_dict,
                                                      self.content_weight,
                                                      self.style_weight)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values
    
    
base_image,style_image = [np.array(im) for im in [base_image,style_image]]
base_channel.send(x = time.time(),y = neptune_image(base_image,
                                                    description = "Image you want to style neuraly"
                                                   )
                 )
style_channel.send(x = time.time(),y = neptune_image(style_image,
                                                     description = "Image that represents the style you want to transfer"
                                                    )
                  )

x = np.random.uniform(0, 255, (1,)+img_size) - 128.

for i in range(STYLE_ITER):
    logging_channel.send(x = time.time(),y = 'Iteration: %s'%i) 
    
    evaluator = Evaluator(layer_dict,CONTENT_WEIGHT,STYLE_WEIGHT)

    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                     fprime=evaluator.grads, maxfun=OPTIM_ITER)
    
    logging_channel.send(x = time.time(),y = 'Current loss value: %s'%min_val)
    loss_channel.send(x = i,y = float(min_val))

    img = utils.deprocess_image(np.copy(x))

    combined_channel.send(x = time.time(),y = neptune_image(raw_image = img,
                                                            description = "your neuraly styled image\nIteration:{}\nContent weight:{}\nStyle weight{}".\
                                                            format(i,CONTENT_WEIGHT,STYLE_WEIGHT)
                                                           )
                         )
