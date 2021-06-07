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
from detectron2.utils.visualizer import ColorMode

from flask import Flask, render_template, request, redirect, url_for, make_response,jsonify
from werkzeug.utils import secure_filename

from visualizer import Visualizer

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
        default="./faster_rcnn_R_50_FPN_3x.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.1,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=['MODEL.WEIGHTS', './model_0049999.pth', 'INPUT.MIN_SIZE_TEST', 3240, 'INPUT.MAX_SIZE_TEST', 9999, 'TEST.DETECTIONS_PER_IMAGE', 4000, 'MODEL.RPN.PRE_NMS_TOPK_TEST', 4000],
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
    transforms.Normalize(mean=[0.098, 0.201, 0.140],
			 std=[0.109, 0.053, 0.170])
    ])


args = get_parser().parse_args()
cfg = setup_cfg(args)
predictor = DefaultPredictor(cfg)

model_ft = models.resnet101(pretrained=False)
num_ftrs = model_ft.fc.in_features
model_ft = nn.Sequential(*list(model_ft.children())[:-1], nn.Sequential(), nn.Conv2d(num_ftrs, 7, 1), nn.Softmax())
model_ft.load_state_dict(torch.load("./checkpoint21.pth"))
model_ft = model_ft.eval()
model_ft = model_ft.cuda(0)

path = "../test_all/mcf7-pfa b-ctn(g) ecad(r)009.png"
ALLOWED_EXTENSIONS = set(['png'])

COLORS = [[0,0,0],
          [1,1,0],
          [1,0,0],
          [1,0,1],
          [0,1,1],
          [0,0,1],
          [0,0.5,0]]

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
        vis0 = Visualizer(img[:,:,::-1], color=0)
        vis1 = Visualizer(img[:,:,::-1], color=1)
        im = io.imread(upload_path)
        mask = np.zeros_like(im)
        predictions = predictor(img)
        boxes = predictions['instances'].pred_boxes.tensor.cpu().numpy()
        det_scores = predictions['instances'].scores.cpu().numpy()

        name_str = str(time.time())
        scores = []
        colors = []
        count = [0,0,0,0,0,0,0]
        folder_name = 'output/'+name_str
        os.mkdir(folder_name)
        os.mkdir(os.path.join(folder_name, 'cells_a'))
        os.mkdir(os.path.join(folder_name, 'cells_b'))
        os.mkdir(os.path.join(folder_name, 'cells_c'))
        os.mkdir(os.path.join(folder_name, 'cells_d'))
        os.mkdir(os.path.join(folder_name, 'cells_e'))
        os.mkdir(os.path.join(folder_name, 'cells_f'))
        os.mkdir(os.path.join(folder_name, 'cells_non'))
        os.mkdir(os.path.join(folder_name, 'det'))
        for dscore, box in zip(det_scores, boxes):
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
            input = torch.unsqueeze(classification_transform(im0), 0).cuda()
            r0 = model_ft(input).squeeze_().detach().cpu().numpy()
            r1 = model_ft(torch.flip(input, [2])).squeeze_().detach().cpu().numpy()
            r2 = model_ft(torch.flip(input, [3])).squeeze_().detach().cpu().numpy()
            r3 = model_ft(torch.flip(input, [2,3])).squeeze_().detach().cpu().numpy()
            x = (r0+r1+r2+r3) / 4
            q = np.argmax(x[1:])+1
            scores.append(str(x[q] if x[q]>0.01 else 0)[:4])
            if x[q] > thres:
                colors.append(COLORS[q])
                mask[y0:y1,x0:x1] = im[y0:y1,x0:x1]
                cv2.imwrite(os.path.join(folder_name, 'cells_%s'%('abcdef'[q-1]), '%.2f_%d_%d.png'%(x[q].item(), y0, x0)), im[y0:y1,x0:x1,::-1])
                count[q] += 1
            else:
                colors.append(COLORS[0])
                cv2.imwrite(os.path.join(folder_name, 'cells_non', '%.2f_%d_%d.png'%(x[q].item(), y0, x0)), im[y0:y1,x0:x1,::-1])
                count[0] += 1
            cv2.imwrite(os.path.join(folder_name, 'det', '%.2f_%d_%d.png'%(dscore, y0, x0)), im[y0:y1,x0:x1,::-1])
        os.system("cp %s %s"%(upload_path, os.path.join(folder_name, 'original.png')))
        vis0.draw_instance_predictions(boxes, scores, colors)
        vis0.get_output().save('static/images/'+name_str+'_cls.png')
        to_text = cv2.imread('static/images/'+name_str+'_cls.png')[:,:,::-1]
        h,w,_ = to_text.shape
        to_text = np.concatenate([np.ones([256,w,3], np.uint8)*128, to_text], axis=0)
        cv2.putText(to_text, "non-CIC: %d"%count[0], (0,50), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (int(COLORS[0][0]*255), int(COLORS[0][1]*255), int(COLORS[0][2]*255)), 2, cv2.LINE_AA)
        cv2.putText(to_text, "a_partial: %d"%count[1], (800,50), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (int(COLORS[1][0]*255), int(COLORS[1][1]*255), int(COLORS[1][2]*255)), 2, cv2.LINE_AA)
        cv2.putText(to_text, "b_one-in-one: %d"%count[2], (0,100), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (int(COLORS[2][0]*255), int(COLORS[2][1]*255), int(COLORS[2][2]*255)), 2, cv2.LINE_AA)
        cv2.putText(to_text, "c_two-in-one: %d"%count[3], (800,100), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (int(COLORS[3][0]*255), int(COLORS[3][1]*255), int(COLORS[3][2]*255)), 2, cv2.LINE_AA)
        cv2.putText(to_text, "d_in turn: %d"%count[4], (0,150), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (int(COLORS[4][0]*255), int(COLORS[4][1]*255), int(COLORS[4][2]*255)), 2, cv2.LINE_AA)
        cv2.putText(to_text, "e_complicated: %d"%count[5], (800,150), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (int(COLORS[5][0]*255), int(COLORS[5][1]*255), int(COLORS[5][2]*255)), 2, cv2.LINE_AA)
        cv2.putText(to_text, "f_unclear_move: %d"%count[6], (0,200), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (int(COLORS[6][0]*255), int(COLORS[6][1]*255), int(COLORS[6][2]*255)), 2, cv2.LINE_AA)
        cv2.imwrite('static/images/'+name_str+'_cls.png', to_text[:,:,::-1])
        cv2.imwrite(os.path.join(folder_name, 'prediction.png'), to_text[:,:,::-1])

        vis1.draw_instance_predictions(boxes, det_scores)
        vis1.get_output().save('static/images/'+name_str+'_det.png')
        vis1.get_output().save(os.path.join(folder_name, 'det.png'))
        io.imsave(os.path.join(basepath, 'static/images', name_str+'_mask.png'), mask)

        return render_template('upload_ok.html', result_name=name_str, input_name=secure_filename(f.filename))

    return render_template('upload.html')


if __name__ == '__main__':
    # app.debug = True
    app.run(host='0.0.0.0', port=6789, debug=True)

