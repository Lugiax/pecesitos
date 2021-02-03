import cv2
import os, sys
import argparse
import time
import numpy as np
import tensorflow as tf
from glob import glob
from PIL import Image
from skimage.morphology import binary_opening, diamond


from herramientas.general import adjustFrame, obtener_frame
#sys.path.append(os.path.abspath('modelos'))
#print('path desde app principal', sys.path)
from modelos import localizador, mascaraNCA


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
                    help='Directorio base donde se descargarán todos los archivos',
                    default='')
parser.add_argument('--frame0', type=str,
                    help='Cuadro de inicio de los videos, puede ser una expresión.', default='0')
parser.add_argument('--frame_max', type=str,
                    help='Número de cuadros máximos a considerar. Puede ser una expresión, cuando es 0 se utilizan todos los cuadros disponibles',
                    default='0')
parser.add_argument('--pesos_loc', type=str,
                    help='Ruta a los pesos del localizador',
                    default='deteccion/pesos/yolo_medium.pt')
parser.add_argument('--pesos_nca', type=str,
                    help='Ruta a los pesos del localizador',
                    default='corridas/NCA90/weights')
parser.add_argument('--no_mostrar', action='store_true',
                    help='Bandera para no mostrar los resultados, se almacenarán en la carpeta correspondiente')

args = parser.parse_args()


print('Se inicia el localizador... ', end='')
loc = localizador.Localizador(os.path.abspath(args.pesos_loc))
print('Iniciado :D')
print('Se inicia el generador de máscaras... ', end='')
nca = mascaraNCA.NCA()
nca.cargar_pesos(args.pesos_nca)
print('Iniciado :D')

###Funciones de apoyo ----------------------------------------------------------------------
def dibujar_rois(img, rois, conf_min=0.5):
    for x1, y1, x2, y2, conf in rois:
        if conf>conf_min:
            img = cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0))
            img = cv2.putText(img, f'{conf:.4}', (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),3)
    return img

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
###-----------------------------------------------------------------------------------------

dir_videos = os.path.abspath(args.data_dir)
save_dir = os.path.join(dir_videos, 'res_sis') if args.save_dir=='' else args.save_dir
p_vid_izq = glob(os.path.join(dir_videos, 'izq*.MP4'))[0]
p_vid_der = glob(os.path.join(dir_videos, 'der*.MP4'))[0]
p_calib_data = os.path.join(dir_videos, 'calib', 'calibData.pk')

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f'Se ha creado el directorio de guardado: {save_dir}')

video_delay = 1
FPS = 25
img_scale = 0.4
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

pausa = not args.no_mostrar
detectar =  args.no_mostrar
while cam_izq.isOpened() or cam_der.isOpened():
    #print(f'Procesando cuadro {f_count}')
    key_pressed = cv2.waitKey(1)
    if key_pressed == ord('q'):
        break
    elif key_pressed == ord(' '):
        pausa = not pausa
    elif key_pressed == ord('d'):
        detectar = not detectar
    elif not pausa:
        frame_izq, error_frames = obtener_frame(cam_izq)
        frame_counter_izq += error_frames
        frame_der, error_frames = obtener_frame(cam_der)
        frame_counter_der += error_frames

    if detectar:
        regs_izq = loc.localizar(frame_izq)
        regs_der = loc.localizar(frame_der)
        frame_izq = dibujar_masks(frame_izq, regs_izq)
        frame_der = dibujar_masks(frame_der, regs_der)



    time.sleep(video_delay)
    img_comp = cv2.hconcat([frame_izq, frame_der])
    img_comp = adjustFrame(img_comp, scale = img_scale)
    if not args.no_mostrar:
        #print('Mostrando')
        cv2.imshow('Frame', img_comp)
    else:
        print(f'Guardando cuadro procesado {f_count}')
        img = Image.fromarray(cv2.cvtColor(img_comp, cv2.COLOR_BGR2RGB))
        img.save(os.path.join(save_dir, f'frame_{f_count}.png'))
    f_count += 1
    if frame_max>0 and (f_count-f0)>=frame_max:
        print('Saliendo...')
        break

cam_izq.release()
cam_der.release()
if not args.no_mostrar: cv2.destroyAllWindows()
