# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import math

import cv2
import numpy as np
import paddle

from paddleseg import utils
from paddleseg.core import infer
from paddleseg.utils import logger, progbar, visualize


def mkdir(path):
    sub_dir = os.path.dirname(path)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)


def partition_list(arr, m):
    """split the list 'arr' into m pieces"""
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]
def rgb2bgr(img):
    r, g, b = cv2.split(img)
    img = cv2.merge([b, g, r])
    return img

def gradient_descent_optim_attack(model, im):
   
    epsilon_ball = 0.1
    steps = 300
    epsilon_stepsize = 1e-2 
    
    shape = im.shape
    
    delta_init = paddle.fluid.initializer.Normal(loc=0.0, scale=0.05)
    delta  = paddle.fluid.layers.create_parameter(im.shape, 'float32', default_initializer=delta_init)
    opt = paddle.optimizer.Adam(learning_rate= 1e-3, parameters = [delta])
    for step in range(steps):
    logits = model(im + delta)

    index = paddle.argmax(logits[0], axis=1, keepdim=True, dtype='int32')
    mask = paddle.fluid.layers.cast(index, bool)
    masked_logit_pos = paddle.masked_select(logits[0][:,1:, :, :], mask)
    masked_logit_neg = paddle.masked_select(logits[0][:,0:1, :, :], mask)
    #print(paddle.fluid.layers.reduce_max(masked_logit_neg), paddle.fluid.layers.reduce_max(masked_logit_pos))
    #loss =  paddle.fluid.layers.reduce_max(masked_logit_pos) - paddle.fluid.layers.reduce_max(masked_logit_neg)
    loss = paddle.fluid.layers.reduce_mean(masked_logit_pos) - paddle.fluid.layers.reduce_mean(masked_logit_neg)
    loss.backward(retain_graph = True)
    opt.minimize(loss)
    print('step:', step,'loss: ', loss)
    delta = paddle.clip(delta, -epsilon_ball, epsilon_ball)
    adv_img = im + delta
    
    
    mean=(0.5, 0.5, 0.5)
    std=(0.5, 0.5, 0.5)
    mean = np.array(mean)[np.newaxis, np.newaxis, :]
    std = np.array(std)[np.newaxis, np.newaxis, :]
    adv_data = np.transpose(adv_img.squeeze(0).detach().cpu().numpy(), (1, 2, 0)) # 400, 400, 3
    adv_data *= std
    adv_data += mean
    adv_data = np.clip(adv_data, 0., 1.)
    adv_data *= 255.
    adv_data = adv_data.astype(np.uint8)
    adv_data = rgb2bgr(adv_data)
    cv2.imwrite('./output/result/adv1.jpg', adv_data) 

    return adv_img      



def gradient_descent_iter_attack(model, im):
   
    epsilon_ball = 0.01
    steps = 50
    epsilon_stepsize = 5e-3 
    
    shape = im.shape
    
    adv_img = paddle.to_tensor(im, dtype='float32', place = paddle.CUDAPlace(0))
        
    
    for step in range(steps):
        adv_img.stop_gradient = False
        logits = model(adv_img)
        #logit = softmax(paddle.fluid.layers.cast(logits[0], paddle.float64))
        #loss = paddle.fluid.layers.reduce_mean(- (target_label * paddle.log(logit) + (1 - target_label) * paddle.log(1 - logit)))           
        index = paddle.argmax(logits[0], axis=1, keepdim=True, dtype='int32')
        mask = paddle.fluid.layers.cast(index, bool)
        masked_logit_pos = paddle.masked_select(logits[0][:,1:, :, :], mask)
        masked_logit_neg = paddle.masked_select(logits[0][:,0:1, :, :], mask)

        loss = paddle.fluid.layers.reduce_mean(masked_logit_neg) - paddle.fluid.layers.reduce_mean(masked_logit_pos)
        print('step:', step,'loss: ', loss)
        loss.backward()
        gradient = adv_img.grad
        if gradient.isnan().any():
            paddle.assign(0.001 * paddle.randn(gradient.shape), gradient)

        normalized_gradient = paddle.sign(gradient)
        eta = epsilon_stepsize * normalized_gradient
        adv_img = adv_img.detach() + eta.detach()
        eta = paddle.clip(adv_img - im, -epsilon_ball, epsilon_ball)
        adv_img = im + eta

    mean=(0.5, 0.5, 0.5)
    std=(0.5, 0.5, 0.5)
    mean = np.array(mean)[np.newaxis, np.newaxis, :]
    std = np.array(std)[np.newaxis, np.newaxis, :]
    adv_data = np.transpose(adv_img.squeeze(0).detach().cpu().numpy(), (1, 2, 0)) # 400, 400, 3
    adv_data *= std
    adv_data += mean
    adv_data = np.clip(adv_data, 0., 1.)
    adv_data *= 255.
    adv_data = adv_data.astype(np.uint8)
    adv_data = rgb2bgr(adv_data)
    cv2.imwrite('./output/result/adv1.jpg', adv_data) 

    return adv_img      


def predict(model,
            model_path,
            transforms,
            image_list,
            image_dir=None,
            save_dir='output',
            aug_pred=False,
            scales=1.0,
            flip_horizontal=True,
            flip_vertical=False,
            is_slide=False,
            stride=None,
            crop_size=None,
            custom_color=None):
    """
    predict and visualize the image_list.

    Args:
        model (nn.Layer): Used to predict for input image.
        model_path (str): The path of pretrained model.
        transforms (transform.Compose): Preprocess for input image.
        image_list (list): A list of image path to be predicted.
        image_dir (str, optional): The root directory of the images predicted. Default: None.
        save_dir (str, optional): The directory to save the visualized results. Default: 'output'.
        aug_pred (bool, optional): Whether to use mulit-scales and flip augment for predition. Default: False.
        scales (list|float, optional): Scales for augment. It is valid when `aug_pred` is True. Default: 1.0.
        flip_horizontal (bool, optional): Whether to use flip horizontally augment. It is valid when `aug_pred` is True. Default: True.
        flip_vertical (bool, optional): Whether to use flip vertically augment. It is valid when `aug_pred` is True. Default: False.
        is_slide (bool, optional): Whether to predict by sliding window. Default: False.
        stride (tuple|list, optional): The stride of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        crop_size (tuple|list, optional):  The crop size of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        custom_color (list, optional): Save images with a custom color map. Default: None, use paddleseg's default color map.

    """
    utils.utils.load_entire_model(model, model_path)
    nranks = paddle.distributed.get_world_size()
    local_rank = paddle.distributed.get_rank()
    if nranks > 1:
        img_lists = partition_list(image_list, nranks)
    else:
        img_lists = [image_list]

    added_saved_dir = os.path.join(save_dir, 'added_prediction')
    pred_saved_dir = os.path.join(save_dir, 'pseudo_color_prediction')
    
    logger.info("Start to predict...")
    progbar_pred = progbar.Progbar(target=len(img_lists[0]), verbose=1)
    color_map = visualize.get_color_map_list(256, custom_color=custom_color)
    
    #with paddle.no_grad():
        
    for param in model.parameters():
        param.stop_gradient = True
        for module in model.sublayers():
            if isinstance(module, (paddle.nn.BatchNorm, paddle.nn.BatchNorm1D,
                                   paddle.nn.BatchNorm2D, paddle.nn.BatchNorm3D)): # if test, add "paddle.nn.Dropout"
                # print("evaled!!")
                module.eval()
    for i, im_path in enumerate(img_lists[local_rank]):
        im = cv2.imread(im_path)
        #print(im)
        ori_shape = im.shape[:2]
        im, _ = transforms(im)
        #print(im.max(), im.min())
        im = im[np.newaxis, ...]
        im = paddle.to_tensor(im)
        adv_img = gradient_descent_iter_attack(model, im)  
        adv_img = gradient_descent_optim_attack(model, im)
        
        if aug_pred:
            pred, _  = infer.aug_inference(
                    model,
                    adv_img,
                    ori_shape=ori_shape,
                    transforms=transforms.transforms,
                    scales=scales,
                    flip_horizontal=flip_horizontal,
                    flip_vertical=flip_vertical,
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size)
        else:
            pred, _ = infer.inference(
                    model,
                    adv_img,
                    ori_shape=ori_shape,
                    transforms=transforms.transforms,
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size)
            
        pred = paddle.squeeze(pred)
        pred = pred.numpy().astype('uint8')
        print((np.argwhere(pred == 1)).shape, (np.argwhere(pred == 0)).shape)
        # get the saved name
        if image_dir is not None:
            im_file = im_path.replace(image_dir, '')
        else:
            im_file = os.path.basename(im_path)
        if im_file[0] == '/' or im_file[0] == '\\':
            im_file = im_file[1:]

        # save added image
        added_image = utils.visualize.visualize(
                im_path, pred, color_map, weight=0.6)
        added_image_path = os.path.join(added_saved_dir, im_file)
        mkdir(added_image_path)
        cv2.imwrite(added_image_path, added_image)
        #print('im==========', im.shape, pred.shape)
        # save pseudo color prediction
        pred_mask = utils.visualize.get_pseudo_color_map(pred, color_map)
        pred_saved_path = os.path.join(
                pred_saved_dir,
                os.path.splitext(im_file)[0] + ".png")
        mkdir(pred_saved_path)
        pred_mask.save(pred_saved_path)

        # pred_im = utils.visualize(im_path, pred, weight=0.0)
        # pred_saved_path = os.path.join(pred_saved_dir, im_file)
        # mkdir(pred_saved_path)
        # cv2.imwrite(pred_saved_path, pred_im)

        progbar_pred.update(i + 1)
