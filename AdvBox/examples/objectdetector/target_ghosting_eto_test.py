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

def run_single_image(FLAGS, cfg):
    
    
    draw_threshold = FLAGS.draw_threshold
    
    data0, datainfo0  = _image2outs(FLAGS.infer_dir, FLAGS.infer_img, cfg)
   
   
    trainer = Trainer(cfg, mode='test')
    trainer.load_weights(cfg.weights)
 
    trainer.model.eval()
    outs0 = trainer.model(data0)
    _draw_result_and_save(FLAGS.infer_img, outs0, data0, datainfo0, draw_threshold)


       
def run(FLAGS, cfg):
  
    # init depre
    depre_settings = {'ImPermute': {},
                      'DenormalizeImage': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225],
                                           'input_channel_axis': 2, 'is_scale': True},
                      'Resize': {'target_size': (638, 850, 3), 'keep_ratio': False, 'interp': 2},
                      'Encode': {}
                      }
    depreprocessor = OperatorCompose(depre_settings)
    draw_threshold = FLAGS.draw_threshold
    dataset_dir = './out_adv/'
    trainer = Trainer(cfg, mode='test')
    trainer.load_weights(cfg.weights)
    trainer.model.eval()
    attack_succ = 0 
    image_filenames = [dataset_dir+ x for x in os.listdir(dataset_dir)] 
    length = len(image_filenames)
    for image_filename in image_filenames:
        data0, datainfo0  = _image2outs(FLAGS.infer_dir, image_filename, cfg)    
        outs0 = trainer.model(data0)
        flag = _obtain_result(FLAGS.infer_img, outs0, data0, datainfo0, draw_threshold)           
        if flag:
            attack_succ += 1
            print('attack_success=====', attack_succ)
    print('len========', length)

def _obtain_result(image_path, outs, data, datainfo, draw_threshold):
    clsid2catid = datainfo['clsid2catid']
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
        bbox_res = batch_res['bbox'][start:end] \
            if 'bbox' in batch_res else None
        for dt in numpy.array(bbox_res):
            if im_id != dt['image_id']:
                continue
            catid, bbox, score = dt['category_id'], dt['bbox'], dt['score']
            #if catid == 3:  
            #    print(catid, '======', score)
            if catid==3 and score > 0.6:
                return False
    return True



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

    #model.eval()
    # loader 中有1张图像，step-id=0，data包含bbox, bbox_num, neck_feats
    for step_id, data in enumerate(loader):
        #output = model(data)
        print('step_id===', step_id)
        
    return data, datainfo 


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
    #run(FLAGS, cfg)
    run_single_image(FLAGS, cfg)

if __name__ == '__main__':
    test()
