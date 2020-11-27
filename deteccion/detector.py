from utils import LectorVideo, extraer_rois, determinar_cabeza_aleta,\
					obtener_puntos, dibujar_rois, emparejar_rois, obtener_region_punto
from skimage.draw import circle
from skimage.color import gray2rgb
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import argparse
import pickle as pk



parser = argparse.ArgumentParser()
parser.add_argument('-vi', '--video_izq', type=str)
parser.add_argument('-oi', '--offset_izq', type=int, default=0)
parser.add_argument('-vd', '--video_der', type=str)
parser.add_argument('-od', '--offset_der', type=int, default=0)
parser.add_argument('-s', '--guardar_frames', type=bool, default=False)
parser.add_argument('-o', '--resultados', type=str)
args = parser.parse_args().__dict__



ROOT_DIR = os.path.abspath(".")
W_MODEL_PATH = os.path.join(ROOT_DIR, "logs/mask_rcnn_fish_0072.h5")
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
RES_DIR = os.path.join(ROOT_DIR, args['resultados'])

sys.path.append("../Mask_RCNN/pycocotools/PythonAPI/pycocotools")
sys.path.append("../Mask_RCNN/mrcnn")
sys.path.append(ROOT_DIR)

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
import coco
import fishvsall
CONFIG = fishvsall.FishConfig()

class InferenceConfig(CONFIG.__class__):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1+1

    DETECTION_MIN_CONFIDENCE = 0.98

class Detector:
    def __init__(self):
        self.config = InferenceConfig()
        self.config.display()

        # Create model object in inference mode.
        self.model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=self.config)

        # Load weights trained on MS-COCO
        self.model.load_weights(W_MODEL_PATH, by_name=True)


    def detectar(self, imgs):
        assert isinstance(imgs, list), 'Las imágenes deben ser pasadas en una lista'

        # Run detection
        res = []
        for im in imgs:
            res.append(self.model.detect([im], verbose=1))

        return res

    def visualizar(self, imgs, res):
        class_names = ['BG', 'fish']    
        # Visualize results
        for image, r in zip(imgs, res):
            visualize.display_instances(image, r[0]['rois'], r[0]['masks'], r[0]['class_ids'], 
                						class_names, r[0]['scores'])


video1_path = os.path.join(ROOT_DIR, args['video_izq'])
video2_path = os.path.join(ROOT_DIR, args['video_der'])
detecciones_vid1 = []
detecciones_vid2 = []

video1 = LectorVideo(video1_path, offset=args['offset_izq'])
video2 = LectorVideo(video2_path, offset=args['offset_der'])

detector = Detector()
r_disk, c_disk = circle(3, 3, 4)

contador = 0
while video1.video.isOpened() and video2.video.isOpened():
    #print(f'Frames {i+1}')
    frame1 = video1.siguiente()
    frame2 = video2.siguiente()

    deteccion1 = detector.detectar([frame1])[0][0]
    deteccion2 = detector.detectar([frame2])[0][0]
    detecciones_vid1.append(deteccion1)
    detecciones_vid2.append(deteccion2)

    if not args['guardar_frames']:
        continue

    img1 = dibujar_rois(frame1, deteccion1)
    img2 = dibujar_rois(frame2, deteccion2)
    emparejamientos = emparejar_rois(frame1, frame2, deteccion1, deteccion2,
                                    max_ratio=0.8)
    ### Se verifica qué imagen tiene la menor cantidad de 
    fig = plt.figure(constrained_layout = True, figsize=(20,16))
    widths = [3,3]
    heights = [4,1,1,1,1]
    spec = fig.add_gridspec(ncols=2, nrows=5, width_ratios=widths,
                            height_ratios=heights)

    video_ax = fig.add_subplot(spec[0,:])
    video_ax.imshow(np.hstack((img1,img2)))
    video_ax.axis('off')

    #peces_axes = []
    for j, rois in enumerate(emparejamientos):
        #print(f'\nRegion {j+1}')
        reg1_ax = fig.add_subplot(spec[j+1,0])
        reg2_ax = fig.add_subplot(spec[j+1,1])
        reg1_ax.axis('off')
        reg2_ax.axis('off')

        reg1 = extraer_rois(frame1, [rois[0]])[0]
        reg2 = extraer_rois(frame2, [rois[1]])[0]
        pun1, int1, dibujos1 = obtener_puntos(reg1, sigma=2.5, devolver_imagenes=True)
        pun2, int2, dibujos2 = obtener_puntos(reg2, sigma=2.5, devolver_imagenes=True)
        skel1 = gray2rgb(rescale_intensity(np.asarray(dibujos1['esqueleto']).astype(int),(0,255)))
        skel2 = gray2rgb(rescale_intensity(np.asarray(dibujos2['esqueleto']).astype(int),(0,255)))

        reg1 = np.clip(reg1+skel1, 0, 255)
        reg2 = np.clip(reg2+skel2, 0, 255)
        pcab1, aleta1 = determinar_cabeza_aleta(pun1, int1)
        pcab2, aleta2 = determinar_cabeza_aleta(pun2, int2)
        
        y01, _, x01, _ = obtener_region_punto(pcab1, reg1.shape, t_ventana = 8)
        reg1[y01+r_disk, x01+c_disk] = [255,0,0]
        y02, _, x02, _ = obtener_region_punto(pcab2, reg2.shape, t_ventana = 8)
        reg2[y02+r_disk, x02+c_disk] = [255,0,0]

        coord_roi1 = rois[0][:2]
        reg1_ax.imshow(reg1)
        reg1_ax.set_title(f'Cabeza: {coord_roi1+pcab1}. Aleta: {coord_roi1+aleta1}')

        coord_roi2 = rois[1][:2]
        reg2_ax.imshow(reg2)
        reg2_ax.set_title(f'Cabeza: {coord_roi2+pcab2}. Aleta: {coord_roi2+aleta2}')

    plt.savefig(os.path.join(RES_DIR,f'cuadro{contador}.png'))
    contador+=1

with open(os.path.join(RES_DIR,'detecciones_vid1.pk'), 'wb') as f:
    pk.dump(detecciones_vid1, f)

with open(os.path.join(RES_DIR,'detecciones_vid2.pk'), 'wb') as f:
    pk.dump(detecciones_vid2, f)
