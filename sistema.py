import cv2
import os, sys
import argparse
import time
import pickle
import csv
import numpy as np
import tensorflow as tf
from glob import glob
from PIL import Image
from skimage.morphology import binary_opening, diamond
from scipy.spatial.distance import euclidean
from numpy.linalg import norm


from herramientas.general import adjustFrame, obtener_frame, emparejar_rois, Grabador,\
                                EstimadorRansac
#sys.path.append(os.path.abspath('modelos'))
#print('path desde app principal', sys.path)
from modelos import localizador#, mascaraNCA


parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str,
                    help='Directorio base donde se encuentran las imágenes y las máscaras')
parser.add_argument('--offset_i', type=int,
                    help='Cuadros de diferencia del video izquierdo',
                    default=0)
parser.add_argument('--offset_d', type=int,
                    help='Cuadros de diferencia del video derecho',
                    default=0)
parser.add_argument('--save_dir', type=str,
                    help='Directorio donde se guardarán todos los archivos generados',
                    default='')
parser.add_argument('--frame0', type=str,
                    help='Cuadro de inicio de los videos, puede ser una expresión.', default='0')
parser.add_argument('--frame_max', type=str,
                    help='Número de cuadros máximos a considerar. Puede ser una expresión, cuando es 0 se utilizan todos los cuadros disponibles',
                    default='0')
parser.add_argument('--pesos_loc', type=str,
                    help='Ruta a los pesos del localizador',
                    default='pesos/yolo/best_large.pt')
parser.add_argument('--pesos_nca', type=str,
                    help='Ruta a los pesos del localizador',
                    default='pesos/nca/weights')
parser.add_argument('--calib_path', type=str,
                    help='Ruta al archivo *.pk de calibración',
                    default='calib/calibData.pk')
parser.add_argument('--no_mostrar', action='store_true',
                    help='Bandera para no mostrar los resultados, se almacenarán en la carpeta correspondiente')

args = parser.parse_args()


print(f'Se inicia el localizador, pesos en {os.path.abspath(args.pesos_loc)}... ', end='')
loc = localizador.Localizador(os.path.abspath(args.pesos_loc), conf_thres=0.8)
print('Iniciado :D')
#print('Se inicia el generador de máscaras... ', end='')
#nca = mascaraNCA.NCA()
#nca.cargar_pesos(args.pesos_nca)
#print('Iniciado :D')

###Funciones de apoyo ----------------------------------------------------------------------
def dibujar_rois(img, rois, color=(255,0,0), conf_min=0.5, txt=''):
    for x1, y1, x2, y2, conf in rois:
        if conf>conf_min:
            img = cv2.rectangle(img, (x1,y1), (x2,y2), color)
            img = cv2.putText(img, f'{conf:.4} {txt}', (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    return img

def puntos_medios(roi):
    x1, y1, x2, y2, _ = roi
    y = (y2+y1)//2
    return (y, x1), (y, x2)

def angulo(p1, p2):
    """
    Devuelve el valor del ángulo del vector p2-p1 considerando
    solamente los ejex x y z. El valor devuelto está en radianes
    """
    p1_2d_y = np.array([p1[0], p1[2]]) 
    p2_2d_y = np.array([p2[0], p2[2]])

    unitario = np.array([1,0])
    diff = p2_2d_y-p1_2d_y
    return np.arccos( np.dot(unitario,diff) / (norm(unitario)*norm(diff)))

"""
def dibujar_masks(img, rois, conf_min=0.5, umbral_mask=0.3):
    canvas = img.copy()
    for x1, y1, x2, y2, conf in rois:
        if conf>conf_min:
            region = img[y1:y2, x1:x2]
            mascara = nca.generar(region, not_bin=False, umbral=0.3, get_original_size=False)#.astype(region.dtype)
            mascara = tf.image.resize(binary_opening(mascara, diamond(3))[None, ..., None].astype(region.dtype),
                                      (y2-y1, x2-x1),
                                      method='bicubic')[0,...,0].numpy()
            mascara = np.clip(mascara, 0, 1)
            mascara_bgr = cv2.cvtColor(mascara, cv2.COLOR_GRAY2BGR)
            mascara_bgr[..., 0] *= 96
            mascara_bgr[..., 1] *= 240
            mascara_bgr[..., 2] *= 91
            canvas = cv2.rectangle(canvas, (x1,y1), (x2,y2), (255,0,0))
            canvas = cv2.putText(canvas, f'{conf:.4}', (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),3)
            canvas[y1:y2, x1:x2, :] = (canvas[y1:y2, x1:x2]*0.3+mascara_bgr*0.7).astype(canvas.dtype)
    return canvas
"""

class StereoEstimator:
    def __init__(self, calib_data_dir, img_shape):       
        with open(calib_data_dir, 'rb') as f:
            self.stereo_params = pickle.load(f)

        ##Se hace el cálculo de la rectificación
        self.R1, self.R2, self.P1, self.P2, _, _, _ =\
                    cv2.stereoRectify(self.stereo_params['cameraMatrix1'],
                                    self.stereo_params['distCoeffs1'],
                                    self.stereo_params['cameraMatrix2'],
                                    self.stereo_params['distCoeffs2'],
                                    img_shape,#img1_o.shape[:2][::-1],
                                    self.stereo_params['R'],
                                    self.stereo_params['T'],
                                    flags=cv2.CALIB_ZERO_DISPARITY,
                                    alpha=0)
    def triangular(self, p1, p2):
        p1_corr = np.array([[p1[::-1]]], dtype=np.float32)
        p2_corr = np.array([[p2[::-1]]], dtype=np.float32)
        corregidos = cv2.correctMatches(self.stereo_params['F'], 
                                        p1_corr, p2_corr)
        p1_corr =  cv2.undistortPoints(corregidos[0], 
                                    self.stereo_params['cameraMatrix1'], 
                                    self.stereo_params['distCoeffs1'],
                                    None, self.R1, self.P1)[0]
        p2_corr =  cv2.undistortPoints(corregidos[1], 
                                    self.stereo_params['cameraMatrix2'], 
                                    self.stereo_params['distCoeffs2'],
                                    None, self.R2, self.P2)[0]
        triangulacion = cv2.convertPointsFromHomogeneous(
                                        cv2.triangulatePoints(self.P1,
                                                            self.P2,
                                                            p1_corr.T,
                                                            p2_corr.T).T
                                                        )[0][0]
        return triangulacion
    
    def distancia(self, p1, p2):
        return euclidean(p1, p2)*24#self.stereo_params.get('tamano_cuadro', 25)
###-----------------------------------------------------------------------------------------
video_delay = 0
FPS = 25
img_scale = 0.4


dir_videos = os.path.abspath(args.data_dir)

save_dir = os.path.join(dir_videos, 'res_sis') if args.save_dir=='' else args.save_dir
p_vid_izq = glob(os.path.join(dir_videos, 'izq*.MP4'))[0]
p_vid_der = glob(os.path.join(dir_videos, 'der*.MP4'))[0]
p_calib_data = args.calib_path

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f'Se ha creado el directorio de guardado: {save_dir}')


frame0 = max(abs(args.offset_d-args.offset_i),
            int(eval(args.frame0)))
#print(frame0)
frame_max = int(eval(args.frame_max))
#print(frame_max)

#print(f'Se muestran los datos {not args.no_mostrar}')
print('Se inician los videos')
cam_izq = cv2.VideoCapture(p_vid_izq)
cam_der = cv2.VideoCapture(p_vid_der)

frame_izq, frame_counter_izq = obtener_frame(cam_izq)
frame_der, frame_counter_der = obtener_frame(cam_der)

frames_offset_izq = args.offset_i + frame0
frames_offset_der = args.offset_d + frame0
f_count = max(frames_offset_izq, frames_offset_der)
f0 = f_count
#Se hace el ajuste de los frames desfasados
print(f'Ajustando frames... {frames_offset_izq}-{frames_offset_der}')
while frames_offset_izq>0 or frames_offset_der>0:
    if frames_offset_izq>0:
        cam_izq.read()
        frames_offset_izq -= 1
        frame_counter_izq += 1
    if frames_offset_der>0:
        cam_der.read()
        frames_offset_der -= 1
        frame_counter_der += 1
print(f'Ajustado :D {frames_offset_izq}-{frames_offset_der}')

#Se inician los estimadores
estimador = StereoEstimator(p_calib_data, img_shape=frame_izq.shape[:2][::-1])
ransac = EstimadorRansac(n_iter=100, p_subset=0.5, p_inliers=0.6)

pausa = not args.no_mostrar
detectar =  args.no_mostrar
grabar = args.no_mostrar
if args.no_mostrar:
    print('Inicio de la grabación')
    grabador_i = Grabador(os.path.join(save_dir, 'grabación_izq.mp4'),FPS)
    grabador_d = Grabador(os.path.join(save_dir, 'grabación_der.mp4'),FPS)

a_escribir = ['frame,p1x,p1y,p1z,p2x,p2y,p2z,distancia,angulo'.split(',')]

buffer = {'long':[],'ang':[]} #Una lista de longitudes y otra de ángulos

while cam_izq.isOpened() or cam_der.isOpened():
    
    #print(f'Procesando cuadro {f_count}')
    key_pressed = cv2.waitKey(1)
    if key_pressed == ord('q'):
        break
    elif key_pressed == ord(' '):
        pausa = not pausa
    elif key_pressed == ord('d'):
        detectar = not detectar
    elif key_pressed == ord('n'):
        frame_izq, error_frames = obtener_frame(cam_izq)
        frame_counter_izq += error_frames
        frame_der, error_frames = obtener_frame(cam_der)
        frame_counter_der += error_frames
    elif not pausa:
        frame_izq, error_frames = obtener_frame(cam_izq)
        frame_counter_izq += error_frames
        frame_der, error_frames = obtener_frame(cam_der)
        frame_counter_der += error_frames

    if detectar:
        rois_izq = loc.localizar(frame_izq)
        rois_der = loc.localizar(frame_der)
        #emparejadas = emparejar_rois(frame_izq, frame_der, rois_izq, rois_der)

        for r1, r2 in zip(rois_izq, rois_der):#enumerate(emparejadas):
            p1_i, p2_i = puntos_medios(r1)
            p1_d, p2_d = puntos_medios(r2)

            p1 = estimador.triangular(p1_i, p1_d)
            p2 = estimador.triangular(p2_i, p2_d)
            buffer['long'].append(estimador.distancia(p1, p2))
            buffer['ang'].append(angulo(p1,p2))
            
            if len(buffer['long'])>50:
                longitud, _ = ransac.estimar(buffer['long'], sigma=0.5)
                if longitud is None:
                    print('Longitud None por ransac')
                    longitud = np.mean(buffer['long'])
            else:
                longitud = np.mean(buffer['long'])    
            
            ang = np.mean(buffer['ang'][-5:])

            frame_izq = dibujar_rois(frame_izq, [r1], txt=f'{longitud:.2f}mm, {ang:.2f}rad')
            frame_der = dibujar_rois(frame_der, [r2], txt=f'{longitud:.2f}mm, {ang:.2f}rad')
            a_escribir.append([f_count]+list(p1)+list(p2)+[longitud, ang])
            #print(a_escribir[-1])
            #print(p1, p2, estimador.distancia(p1, p2))
            #print()



    time.sleep(video_delay)
    img_comp = cv2.hconcat([frame_izq, frame_der])
    img_comp = adjustFrame(img_comp, scale = img_scale)
    if not args.no_mostrar:
        #print('Mostrando')
        cv2.imshow('Frame', img_comp)
    else:
        #img = Image.fromarray(cv2.cvtColor(img_comp, cv2.COLOR_BGR2RGB))
        #img.save(os.path.join(save_dir, f'frame_{f_count}.png'))
        grabador_i.agregar(frame_izq)
        grabador_d.agregar(frame_der)
        print(f'Guardando cuadro procesado {f_count}')

    f_count += 1
    if frame_max>0 and (f_count-f0)>=frame_max:
        print('Saliendo...')
        break

cam_izq.release()
cam_der.release()
if not args.no_mostrar: 
    cv2.destroyAllWindows()
else:
    grabador_i.terminar()
    grabador_d.terminar()
    with open(os.path.join(save_dir, 'grabación.csv'), 'w') as f:
        escritor = csv.writer(f, delimiter=',')
        escritor.writerows(a_escribir)

