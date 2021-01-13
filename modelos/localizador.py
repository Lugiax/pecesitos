import sys
import os

import torch
import numpy as np
from numpy import random

file_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(file_dir, 'yolov5'))

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device



class Localizador:
    def __init__(self, dir_pesos, imgsz=640, device='', conf_thres=0.25, iou_thres=0.45, classes=['0']):
        # Initialize
        self.conf_thres=conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.device = select_device(device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.model = attempt_load(dir_pesos, map_location=self.device)  # load FP32 model
        self.imgsz = check_img_size(imgsz, s=self.model.stride.max())  # check img_size
        if self.half:
            self.model.half()  # to FP16

        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device)  # init img
        _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once

    def localizar(self, im0):
        img = letterbox(im0, new_shape=self.imgsz)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        predictions = self.model(img, augment=False)[0]
        predictions = non_max_suppression(predictions, self.conf_thres, self.iou_thres)[0]
        predictions[:, :4] = scale_coords(img.shape[2:], predictions[:, :4], im0.shape).round()

        res = []
        for pred in predictions:
            pred = pred.tolist()
            res.append([int(n) for n in pred[:4]] + pred[4:5])

        return res


if __name__ == '__main__':
    yolov5_path = os.path.abspath('./yolov5')
    sys.path.append(yolov5_path)
    import cv2
    img_path = '/home/carlos/Documentos/Codes/ProyectoMCC/datos/imagenes/0b91f658f06b594f.jpg'
    img = cv2.imread(img_path)
    cv2.imshow('Original', img)

    pesos_path = '/home/carlos/Documentos/Codes/ProyectoMCC/deteccion/pesos/yolo_medium.pt'
    loc = Localizador(pesos_path)
    res = loc.localizar(img)
    for x1,y1,x2,y2,conf in res:
        cv2.imshow(f'Conf: {conf}', img[y1:y2, x1:x2])
        
    cv2.waitKey(0)
    cv2.destroyAllWindows

