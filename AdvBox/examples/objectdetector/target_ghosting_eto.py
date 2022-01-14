#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The Target Ghosting Attack demonstration.
Contains:
* Initialize a yolo detector and inference pictures.
* Generate perturbation using model weights.
* Generate perturbed image.

Author: xiongjunfeng
"""
# ignore warning log
import warnings
warnings.filterwarnings('ignore')
import glob
import os
import cv2
from PIL import Image
import xmltodict
import paddle
from ppdet.core.workspace import create
from ppdet.core.workspace import load_config, merge_config
from ppdet.engine import Trainer
from ppdet.utils.check import check_gpu, check_version, check_config
from ppdet.utils.cli import ArgsParser
from ppdet.slim import build_slim_model
from ppdet.utils.visualizer import visualize_results, save_result
from ppdet.data.source.category import get_categories
from ppdet.metrics import get_infer_results
from depreprocess.operator_composer import OperatorCompose
import paddle.nn.functional as F
import copy
from EOT_simulation import transformation
import numpy
from ppdet.utils.logger import setup_logger
import math
logger = setup_logger('train')
#paddle.enable_static() 

def parse_args():
    parser = ArgsParser()
    parser.add_argument(
        "--infer_dir",
        type=str,
        default=None,
        help="Directory for images to perform inference on.")
    parser.add_argument(
        "--infer_img",
        type=str,
        default=None,
        help="Image path, has higher priority over --infer_dir")
    parser.add_argument(
        "--target_img",
        type=str,
        default=None,
        help="Image path, infer image with masked on.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory for storing the output visualization files.")
    parser.add_argument(
        "--draw_threshold",
        type=float,
        default=0.5,
        help="Threshold to reserve the result for visualization.")
    parser.add_argument(
        "--slim_config",
        default=None,
        type=str,
        help="Configuration file of slim method.")
    parser.add_argument(
        "--use_vdl",
        type=bool,
        default=False,
        help="Whether to record the data to VisualDL.")
    parser.add_argument(
        '--vdl_log_dir',
        type=str,
        default="vdl_log_dir/image",
        help='VisualDL logging directory for image.')
    parser.add_argument(
        "--save_txt",
        type=bool,
        default=False,
        help="Whether to save inference result in txt.")
    args = parser.parse_args()
    return args


def get_test_images(infer_dir, infer_img):
    """
    Get image path list in TEST mode
    """
    assert infer_img is not None or infer_dir is not None, \
        "--infer_img or --infer_dir should be set"
    assert infer_img is None or os.path.isfile(infer_img), \
            "{} is not a file".format(infer_img)
    assert infer_dir is None or os.path.isdir(infer_dir), \
            "{} is not a directory".format(infer_dir)

    # infer_img has a higher priority
    if infer_img and os.path.isfile(infer_img):
        return [infer_img]

    images = set()
    infer_dir = os.path.abspath(infer_dir)
    assert os.path.isdir(infer_dir), \
        "infer_dir {} is not a directory".format(infer_dir)
    exts = ['jpg', 'jpeg', 'png', 'bmp']
    exts += [ext.upper() for ext in exts]
    for ext in exts:
        images.update(glob.glob('{}/*.{}'.format(infer_dir, ext)))
    images = list(images)

    assert len(images) > 0, "no image found in {}".format(infer_dir)
    logger.info("Found {} inference images in total.".format(len(images)))

    return images


def get_pcls(model, neck_feats):
   
    # assert len(feats) == len(self.anchors)
    pcls_list = []
    #print('neck======', neck_feats)
    for i, feat in enumerate(neck_feats):
        yolo_output = model.yolo_head.yolo_outputs[i](feat)
        
        if model.data_format == 'NHWC':
            yolo_output = paddle.transpose(yolo_output, [0, 3, 1, 2])
        #print('i======', i, feat.shape, yolo_output.shape)
        p = yolo_output
        number_anchor = 3
        b, c, h, w = p.shape

        p = p.reshape((b, number_anchor, -1, h, w)).transpose((0, 1, 3, 4, 2))
        # x, y = p[:, :, :, :, 0:1], p[:, :, :, :, 1:2]
        # w, h = p[:, :, :, :, 2:3], p[:, :, :, :, 3:4]
        #print('p======', p.shape)
        obj, pcls = p[:, :, :, :, 4:5], p[:, :, :, :, 5:]
        #print(obj.shape, pcls.shape)
        pcls_list.append(pcls)

    return pcls_list


def pcls_kldivloss(pcls_list, target_pcls_list):
    """
    Compute the kl distance between pcls and target pcls.
    Args:
        pcls_list: list. Middle output from yolo loss. pcls is the classification feature map.
        target_pcls_list: list. The target pcls feature map.

    Returns:
        paddle.tensor. kl distance.
    """
    kldiv_criterion = paddle.nn.KLDivLoss(reduction='batchmean')
    logsoftmax = paddle.nn.LogSoftmax()
    softmax = paddle.nn.Softmax()
    kldivloss = 0

    for pcls, target_pcls in zip(pcls_list, target_pcls_list):
        loss_kl = kldiv_criterion(logsoftmax(pcls), softmax(target_pcls))
        kldivloss += loss_kl

    return kldivloss




def get_mask_coordination(_object):
    """
    Place mask coordination in variables.

    Args:
    maskfilename: Path for the xml file containing mask coordination.
    **kwargs: Other named arguments.
    """
    xmin = int(_object['bndbox']['xmin'])
    ymin = int(_object['bndbox']['ymin'])
    xmax = int(_object['bndbox']['xmax'])
    ymax = int(_object['bndbox']['ymax'])

    return xmin,ymin,xmax,ymax





class attack_net(paddle.fluid.dygraph.Layer):
    def __init__(self, cfg):
        super(attack_net, self).__init__()
        self.trainer = Trainer(cfg, mode='test')
        self.trainer.load_weights(cfg.weights)
        
        # extrack mask
        f = open('test/EOTB_car.xml')
        dic = xmltodict.parse(f.read())
        mask_list = dic['annotation']['object']
        box_list = dic['annotation']['size']
        widtht, heightt = int(box_list['width']), int(box_list['height'])
        xmin, ymin, xmax, ymax = get_mask_coordination(mask_list[0])
        self.xmin_1 = math.floor(320/widtht * xmin) # floor
        self.ymin_1 = math.floor(320/heightt * ymin) # floor
        self.xmax_1 = math.ceil(320/widtht * xmax) # ceil
        self.ymax_1 = math.ceil(320/heightt * ymax) # ceil
        height = self.ymax_1 - self.ymin_1
        width = self.xmax_1 - self.xmin_1      
        self.init_inter_mask = paddle.fluid.initializer.Normal(loc=0.0, scale=0.75) # [-3, 3]
        self.masked_inter = paddle.fluid.layers.create_parameter([1, 3, height, width], 'float32', name="masked_inter", default_initializer=self.init_inter_mask)
        #print(self.masked_inter.numpy().min(), self.masked_inter.numpy().max())
        # init depre
        EOT_transforms = transformation.target_sample()
        self.num_of_EOT_transforms = len(EOT_transforms)
        self.transform = numpy.array(EOT_transforms).reshape(((94, 2, 3)))
        self.transform = paddle.to_tensor(self.transform, dtype= 'float32') 
        self.nnSigmoid = paddle.nn.Sigmoid()
        #self.nnlogSoftmax = paddle.nn.LogSoftmax()
        #self.nnSoftmax = paddle.nn.Softmax()
    @paddle.no_grad()
    def ext_out(self, input_data, constrained):
        input_data['image'][:, :, self.ymin_1:self.ymax_1, self.xmin_1:self.xmax_1]= constrained[0:1, :, self.ymin_1:self.ymax_1, self.xmin_1:self.xmax_1]
        outs = self.trainer.model(input_data)
        return outs, input_data['image'][0].detach()
    
    def forward(self, input1, input2):
        
        useEOT = True
        if useEOT == True:
            # broadcast self.masked_inter [1,448,448,3] into [num_of_EOT_transforms, 448, 448, 3]
            masked_inter_batch = self.masked_inter
            
            for i in range(self.num_of_EOT_transforms):
                if i == self.num_of_EOT_transforms-1: break
                masked_inter_batch = paddle.concat([masked_inter_batch, self.masked_inter],0)
            

        else:
            masked_inter_batch = self.masked_inter
        
        X = input1['image'].detach()
        #print('min====', X.numpy().min(),'max=====',  X.numpy().max())
        X_batch = X
               
        for i in range(self.num_of_EOT_transforms-1):
            X_batch = paddle.concat([X_batch, X],0) 
        
        masked_inter_batch_val = paddle.clip(masked_inter_batch, min=-5.0, max=5.0) # -3.15, 3.15
        left1 = X_batch[:, :, self.ymin_1:self.ymax_1, 0:self.xmin_1]
        right1 = X_batch[:, :, self.ymin_1:self.ymax_1, self.xmax_1:320]
        bottom1 = X_batch[:, :, 0:self.ymin_1, :]
        top1 = X_batch[:, :, self.ymax_1:320, :]
          
        shuru = paddle.concat([left1, masked_inter_batch_val], axis =3)
        shuru = paddle.concat([shuru, right1], axis =3)
        shuru = paddle.concat([bottom1, shuru], axis=2)
        shuru = paddle.concat([shuru, top1], axis=2)
        grid = paddle.nn.functional.affine_grid(self.transform, shuru.shape)
       
        shuru = paddle.nn.functional.grid_sample(shuru, grid, mode='bilinear')
      
        constrained = shuru
        print(shuru.shape)
      
        constrained[:, :, self.ymin_1:self.ymax_1, self.xmin_1:self.xmax_1] = paddle.tanh(constrained[:, :, self.ymin_1:self.ymax_1, self.xmin_1:self.xmax_1])       
   
        attack_region = X_batch[:, :, self.ymin_1:self.ymax_1, self.xmin_1:self.xmax_1]
        adverse_region = constrained[:, :, self.ymin_1:self.ymax_1, self.xmin_1:self.xmax_1]
        perturbation = attack_region - adverse_region
  
        distance_L21 = paddle.norm(perturbation, p=2, axis=2) # 94, 94, 95, 3->94,95, 3
        distance_L22 = paddle.norm(distance_L21, p=2, axis=2) # 94,3
        distance_L23 = paddle.norm(distance_L22, p=2, axis=1) # 94,
        distance_L2 = paddle.fluid.layers.reduce_mean(distance_L23, dim=0)
    
        lala1 = adverse_region[:, :, 0:-1, 0:-1]
        lala2 = adverse_region[:, :, 1:, 1:]

        sub_lala1_2 = lala1-lala2
        non_smoothness1 = paddle.norm(sub_lala1_2, p=2, axis=2)
        non_smoothness2 = paddle.norm(non_smoothness1, p=2, axis=2)
        non_smoothness3 = paddle.norm(non_smoothness2, p=2, axis=1)
        non_smoothness = paddle.fluid.layers.reduce_mean(non_smoothness3, dim=0)
        input2['image'] = constrained
        self.trainer.model.eval()
        outs2 = self.trainer.model(input2)
       
        pcls_list = get_pcls(self.trainer.model, outs2['neck_feats'])
        C_target = 0.
        C_nontarget = 0.
        for pcls in pcls_list:
            b, anc, h, w, cls = pcls.shape
            pcls = self.nnSigmoid(pcls)
            
            x1 =  pcls[:, :, :, :, 0:3]
            x2 = pcls[:, :, :, :, 5:8]
            x3 = pcls[:, :, :, :, 9:]
            
            x = paddle.concat([x1, x2, x3], axis = -1)
            x = paddle.fluid.layers.reduce_max(x, dim=-1)
            
            x = paddle.reshape(x, [b, anc*h*w])
            x, _ = (paddle.topk(x, 7, axis=1))
            x = paddle.fluid.layers.reduce_sum(x, 1) 
            x = paddle.fluid.layers.reduce_sum(x, 0)
            C_nontarget += x
            
            pcls_3 = paddle.reshape(pcls[:, :, :, :, 3], [b, anc*h*w])
            pcls_3 = paddle.fluid.layers.reduce_max(pcls_3, 1) 
            pcls_3 = paddle.fluid.layers.reduce_sum(pcls_3, 0) # b, 1
            C_target += 0.8*pcls_3
            pcls_4 = paddle.reshape(pcls[:, :, :, :, 4], [b, anc*h*w])
            pcls_4 = paddle.fluid.layers.reduce_max(pcls_4, 1) # b
            pcls_4 = paddle.fluid.layers.reduce_sum(pcls_4, 0) # b, 1
            C_target += 0.6*pcls_4
            pcls_8 = paddle.reshape(pcls[:, :, :, :, 8], [b, anc*h*w])
            pcls_8 = paddle.fluid.layers.reduce_max(pcls_8, 1) # b
            pcls_8 = paddle.fluid.layers.reduce_sum(pcls_8, 0) # b, 1
            C_target += 0.1*pcls_8
            
     
        punishment = 0.001
        smoothness_punishment = 0.05 
    
        loss = (C_nontarget - C_target)/94. + punishment* distance_L2 + smoothness_punishment* non_smoothness
        outs_adv, in_adv = self.ext_out(input1, constrained)
        
        return loss, outs_adv, in_adv



def run(FLAGS, cfg):
    
    # init depre
    depre_settings = {'ImPermute': {},
                      'DenormalizeImage': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225],
                                           'input_channel_axis': 2, 'is_scale': True},
                      'Resize': {'target_size': (533, 800, 3), 'keep_ratio': False, 'interp': 2}, #638, 850, 864, 1152
                      'Encode': {}
                      }
    depreprocessor = OperatorCompose(depre_settings)
    draw_threshold = FLAGS.draw_threshold
  
    
    data0, datainfo0  = _image2outs(FLAGS.infer_dir, FLAGS.infer_img, cfg)
     
    epochs = 800
  
    model_attack = attack_net(cfg)
    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=0.01, T_max=800, verbose=True)
    data, _ = batchloader(FLAGS.infer_dir, FLAGS.infer_img, cfg)
    #print('parameters==========', model_attack.parameters())
    opt = paddle.optimizer.Adam(learning_rate= scheduler, parameters = model_attack.parameters())
    
    
    for epoch in range (epochs):       
        loss, outs_adv, data_adv = model_attack(data0, data) 
        loss.backward()     
        opt.minimize(loss)
        print('Epoch:', epoch, '======loss:', loss.numpy())
        flag = ext_score(outs_adv, data0, datainfo0)
        if flag:
            break
       
    #_draw_result_and_save('car_vis.jpeg', outs0, data0, datainfo0, draw_threshold)
    data_adv = depreprocessor(data_adv)  
   
    cv2.imwrite('adverse_car.jpeg', data_adv)
       
def batchloader(infer_dir, infer_img, cfg):
    # get inference images
    mode = 'test'
    
    dataset = cfg['{}Dataset'.format(mode.capitalize())]
    for i in range(94):
        images = get_test_images(infer_dir, infer_img)
        dataset.set_images(images*(i+1))
    
    imid2path = dataset.get_imid2path
    #print("imid2path===== ", imid2path)
    anno_file = dataset.get_anno()
    #print('anno_file======', anno_file)
    clsid2catid, catid2name = get_categories(cfg.metric, anno_file=anno_file)
    datainfo = {'imid2path': imid2path,
                'clsid2catid': clsid2catid,
                'catid2name': catid2name}
    _eval_batch_sampler = paddle.io.BatchSampler(
                dataset, batch_size=94)
    loader = create('{}Reader'.format(mode.capitalize()))(
                dataset, 2, _eval_batch_sampler)
    for step_id, data in enumerate(loader):
        print('========')
    
    return data, datainfo





def _image2outs(infer_dir, infer_img, cfg):
    # get inference images
    mode = 'test'
    dataset = cfg['{}Dataset'.format(mode.capitalize())]
    
    images = get_test_images(infer_dir, infer_img)
    dataset.set_images(images*1)

    loader = create('TestReader')(dataset, 0)
    imid2path = dataset.get_imid2path
    #print("imid2path===== ", imid2path)
    anno_file = dataset.get_anno()
    #print('anno_file======', anno_file)
    clsid2catid, catid2name = get_categories(cfg.metric, anno_file=anno_file)
    datainfo = {'imid2path': imid2path,
                'clsid2catid': clsid2catid,
                'catid2name': catid2name}

    for step_id, data in enumerate(loader):
        print('step_id===', step_id)
        
    return data, datainfo 


def ext_score(outs, data, datainfo):
    clsid2catid = datainfo['clsid2catid']
    catid2name = datainfo['catid2name']
    for key in ['im_shape', 'scale_factor', 'im_id']:
        outs[key] = data[key]
    for key, value in outs.items():
        if hasattr(value, 'numpy'):
            outs[key] = value.numpy()

    batch_res = get_infer_results(outs, clsid2catid)
    start = 0
    flag1 = False
    flag2 = False
    bbox_num = outs['bbox_num']
    for i, im_id in enumerate(outs['im_id']):
        end = start + bbox_num[i]
        bbox_res = batch_res['bbox'][start:end]
        for dt in numpy.array(bbox_res):
            catid, bbox, score = dt['category_id'], dt['bbox'], dt['score']
            if catid == 3  and score < 0.499:
                print('res=========', catid,  score)
                flag1 = True
            if catid == 8  and score < 0.499:
                flag2 = True
            if catid==3 or catid == 4 or catid ==8:
                print(catid, '===', score)
        if flag1 and flag2:
            return True 
     
    return False

def _draw_result_and_save(image_path, outs, data, datainfo, draw_threshold):
    clsid2catid = datainfo['clsid2catid']
    catid2name = datainfo['catid2name']
    for key in ['im_shape', 'scale_factor', 'im_id']:
        outs[key] = data[key]
    for key, value in outs.items():
        if hasattr(value, 'numpy'):
            outs[key] = value.numpy()

    batch_res = get_infer_results(outs, clsid2catid)
    bbox_num = outs['bbox_num']

    start = 0
    for i, im_id in enumerate(outs['im_id']):
        end = start + bbox_num[i]
        image = Image.open(image_path).convert('RGB')

        bbox_res = batch_res['bbox'][start:end] \
            if 'bbox' in batch_res else None
        mask_res = batch_res['mask'][start:end] \
            if 'mask' in batch_res else None
        segm_res = batch_res['segm'][start:end] \
            if 'segm' in batch_res else None
        keypoint_res = batch_res['keypoint'][start:end] \
            if 'keypoint' in batch_res else None
        image = visualize_results(
            image, bbox_res, mask_res, segm_res, keypoint_res,
            int(im_id), catid2name, draw_threshold)

        # save image with detection
        save_name = os.path.join('output/', 'out_' + os.path.basename(image_path))
        logger.info("Detection bbox results save in {}".format(
            save_name))
        image.save(save_name, quality=95)
        start = end


def test():
    FLAGS = parse_args()
    cfg = load_config(FLAGS.config)
    cfg['use_vdl'] = FLAGS.use_vdl
    cfg['vdl_log_dir'] = FLAGS.vdl_log_dir
    merge_config(FLAGS.opt)

    place = paddle.set_device('gpu' if cfg.use_gpu else 'cpu')

    if 'norm_type' in cfg and cfg['norm_type'] == 'sync_bn' and not cfg.use_gpu:
        cfg['norm_type'] = 'bn'

    if FLAGS.slim_config:
        cfg = build_slim_model(cfg, FLAGS.slim_config, mode='test')

    check_config(cfg)
    check_gpu(cfg.use_gpu)
    check_version()
    run(FLAGS, cfg)


if __name__ == '__main__':
    test()
