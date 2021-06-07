# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import atexit
import bisect
import multiprocessing as mp
from collections import deque
import cv2
import torch
import argparse
import glob
import os
import time
import cv2
import tqdm
import numpy as np
from datetime import timedelta
import torch.nn as nn

from PIL import Image
from skimage import io, transform
from torchvision import models, transforms

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from detectron2.data.detection_utils import read_image
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
from predictor import VisualizationDemo

from flask import Flask, render_template, request, redirect, url_for, make_response,jsonify
from werkzeug.utils import secure_filename

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="../detection_ckpts/output4/faster_rcnn_R_50_FPN_3x.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.2,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=['MODEL.WEIGHTS', '../detection_ckpts/output4/model_0019999.pth', 'INPUT.MIN_SIZE_TEST', 2160, 'INPUT.MAX_SIZE_TEST', 9999, 'TEST.DETECTIONS_PER_IMAGE', 2000],
        nargs=argparse.REMAINDER,
    )
    return parser

class Transform:
    def __init__(self, scale=232):
        self.scale = scale

    @staticmethod
    def cut_border(im, ratio):
        h,w,_ = im.shape
        bb = int(ratio*h)
        im = im[bb:-bb,bb:-bb]
        return im

    def __call__(self, image):
        h,w = image.shape[:2]
        if h > w:
            p0 = (h-w) // 2
            p1 = (h-w) - p0
            image = np.pad(image, [[0,0],[p0,p1],[0,0]], mode='edge')
        else:
            p0 = (w-h) // 2
            p1 = (w-h) - p0
            image = np.pad(image, [[p0,p1],[0,0],[0,0]], mode='edge')
        image = self.cut_border(image, 0.12)
        image = transform.resize(image, [224,224], preserve_range=True).astype(np.uint8)
        return Image.fromarray(image)

classification_transform = transforms.Compose([
    Transform(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
			 std=[0.229,0.224,0.225])
    ])


args = get_parser().parse_args()
cfg = setup_cfg(args)
demo = VisualizationDemo(cfg)
predictor = DefaultPredictor(cfg)

model_ft = models.resnet101(pretrained=False)
num_ftrs = model_ft.fc.in_features
model_ft = nn.Sequential(*list(model_ft.children())[:-1], nn.Conv2d(num_ftrs, 2, 1), nn.Softmax())
model_ft.load_state_dict(torch.load("../classification_ckpt/checkpoint1.pth"))
model_ft = model_ft.eval()
model_ft = model_ft.cuda(0)

path = "../test_all/mcf7-pfa b-ctn(g) ecad(r)009.png"
ALLOWED_EXTENSIONS = set(['png'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
 
app = Flask(__name__)
app.send_file_max_age_default = timedelta(seconds=1)
 
 
# @app.route('/', methods=['POST', 'GET'])
@app.route('/<float:thres>', methods=['POST', 'GET'])
def upload(thres):
    if request.method == 'POST':
        f = request.files['file']
 
        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "仅限于png"})
 
        user_input = request.form.get("name")
 
        basepath = os.path.dirname(__file__)
 
        upload_path = os.path.join(basepath, 'static/images', secure_filename(f.filename)) 
        f.save(upload_path)
 
        img = read_image(upload_path, format="BGR")
        im = io.imread(upload_path)
        mask = np.zeros_like(im)
        predictions, visualized_output = demo.run_on_image(img)
        boxes = predictions['instances'].pred_boxes.tensor.cpu().numpy()
        for box in boxes:
            x0,y0,x1,y1 = box
            h = y1-y0
            w = x1-x0
            x0 -= w*0.16
            x1 += w*0.16
            y0 -= h*0.16
            y1 += h*0.16
            x0 = max(0, x0)
            y0 = max(0, y0)
            x0, x1, y0, y1 = int(x0), int(x1), int(y0), int(y1)
            im0 = im[y0:y1,x0:x1]
            x = model_ft(torch.unsqueeze(classification_transform(im0), 0).cuda()).squeeze_().detach()[1].cpu().numpy().item()
            if x > thres:
                mask[y0:y1,x0:x1] = im[y0:y1,x0:x1]

        name_str = str(time.time())
        visualized_output.save(os.path.join(basepath, 'static/images', "%s_det.png"%name_str))
        io.imsave(os.path.join(basepath, 'static/images', "%s_mask.png"%name_str), mask)
 
        return render_template('upload_ok.html', result_name=name_str, input_name=secure_filename(f.filename))
 
    return render_template('upload.html')
 
 
if __name__ == '__main__':
    # app.debug = True
    app.run(host='0.0.0.0', port=6789, debug=True)
