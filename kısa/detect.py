# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (macOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import sys
import torch
import utils


from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
import numpy as np


class Yolo():
        def __init__(self):

            self.settings = {
                "weights" : 'yolov5s.pt',
                "source":"0",
                "data": 'data/coco128.yaml',
                "imgsz":(640, 640),
                "conf_thres":0.25, 
                "iou_thres":0.45, 
                "max_det":1000, 
                "device":'cuda:0', 
                "view_img":False,
                "save_txt":False,  
                "save_conf":False, 
                "save_crop":False,  
                "nosave":False, 
                "classes":None,
                "agnostic_nms":False,
                "augment":False,
                "visualize":False,
                "update":False,  
                "project": 'runs/detect', 
                "name":'exp',
                "exist_ok":False, 
                "line_thickness":3,
                "hide_labels":False,
                "hide_conf":False,
                "half":False, 
                "dnn":False
            }
            self.device =  self.settings["device"]
            self.model = DetectMultiBackend(self.settings["weights"], device= self.device, dnn=False, data=self.settings["data"], fp16=False)
            self.stride = self.model.stride
            
        def preprocess(self, frame):
            im = cv2.resize(frame,(384,640))
            #imgsz = check_img_size(imgsz, s=self.stride)  # check image size
            im = torch.from_numpy(im).to(self.device)
            im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
            #im /= 255  # 0 - 255 to 0.0 - 1.0

            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim [1, 384, 640,3]
            
            im = np.swapaxes(im,1,3)

            im = np.rollaxis(im, 2,3)
            

            return im

        def postprocess(self, im0, pred, classes):
             # NMS
            
            pred = non_max_suppression(pred, self.settings["conf_thres"], self.settings["iou_thres"], classes, self.settings["agnostic_nms"], max_det=self.settings["max_det"])

            # Second-stage classifier (optional)
            #pred = utils.general.apply_classifier(pred, self.settings["classifier_model"], im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im0.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class

                    # Write results
                    for *xyxy, conf, cls in reversed(det):

                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        print(xywh)
                            

                # Stream results


                return pred

        def detect(self, frame, classes):

            frame = self.preprocess(frame)
            
            result = self.model(frame, augment=False)

            bbox = self.postprocess(frame, result, classes)

            return bbox

   

if __name__ == "__main__":

    
    detector = Yolo()
    img = cv2.imread("data/images/zidane.jpg")

    classes=[0]
    result = detector.detect(img,classes)
    print("Bitti")