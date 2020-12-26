import numpy as np
import cv2
from datetime import date
from glob import glob
import os
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str,
                    help='Directorio base donde se encuentran las imágenes y las máscaras')
parser.add_argument('--offset_i', type=int,
                    help='Cuadros de diferencia del video izquierdo', default=0)
parser.add_argument('--offset_d', type=int,
                    help='Cuadros de diferencia del video derecho', default=0)
parser.add_argument('--save_dir', type=str,
                    help='Directorio base donde se descargarán todos los archivos',
                    default='')
parser.add_argument('--frame0', type=str,
                    help='Cuadro de inicio de los videos, puede ser una expresión.', default='0')
args = parser.parse_args()


pattern_size = (8,6)
videos_dir = args.data_dir
assert os.path.isdir(videos_dir), 'Ingrese un directorio de videos válido'
video_path_izq = glob(os.path.join(videos_dir, 'izq*.MP4'))[0]
video_path_der = glob(os.path.join(videos_dir, 'der*.MP4'))[0]
assert os.path.isfile(video_path_izq) and os.path.isfile(video_path_der),\
        'Archivos de video no válidos'
dir_to_save = os.path.join(videos_dir, 'imgs') if args.save_dir=='' else args.save_dir
video_delay = 0.00
frames_offset_izq = args.offset_i
frames_offset_der = args.offset_d
FPS = 25
img_scale = 0.4
show = True
frame0 = max(abs(frames_offset_der-frames_offset_izq),
            int(eval(args.frame0)))

#---------------------------------------------------------------------------------------

##Si se desea guardar los frames:
def saveFrame(frame, dir_name='',text=''):
    if not os.path.isdir(os.path.join(dir_to_save, dir_name)):
        os.makedirs(os.path.join(dir_to_save, dir_name))
    name = dir_name+f'frame_{text}.png'
    cv2.imwrite(os.path.join(dir_to_save, name), frame)

def frames_to_time(frames):
    segundos = frames//FPS
    tiempo = [str(segundos)]
    frames %= FPS
    if segundos>59:
        minutos = segundos//60
        segundos %= 60
        tiempo = ['{:0>2}'.format(str(minutos)),
                '{:0>2}'.format(str(segundos))]
    return(':'.join(tiempo)+'.{:0>2}'.format(frames))

def obtener_frame(camara, frame=None, ret=False):
    max_frames_error = 50
    error_frames_counter = 0
    while not ret:
        if not ret and error_frames_counter<max_frames_error: 
            error_frames_counter+=1
            ret, frame = camara.read()
        elif error_frames_counter==max_frames_error:
            print('LÍMITE DE FRAMES CON ERROR ALCANZADO. SALIENDO.')
            break
    return frame, error_frames_counter

def adjustFrame(frame, scale = 1):
    H, W = frame.shape[:2]
    return cv2.resize(frame, (int(W*scale), int(H*scale)),
                        interpolation = cv2.INTER_CUBIC)


assert os.path.exists(video_path_izq)
assert os.path.exists(video_path_der)

if not os.path.isdir(dir_to_save):
    os.makedirs(os.path.join(dir_to_save,'izq'))
    os.makedirs(os.path.join(dir_to_save,'der'))
    print(f'Se creó el directorio {dir_to_save}')


## Comienza la captura del video
cam_izq = cv2.VideoCapture(video_path_izq)
cam_der = cv2.VideoCapture(video_path_der)
saved_frames_counter = 1
frame_counter_der = 0
frame_counter_izq = 0
pausa = True
save = False

frame_izq, error_frames = obtener_frame(cam_izq)
frame_counter_izq += error_frames
frame_der, error_frames = obtener_frame(cam_der)
frame_counter_der += error_frames

frames_offset_der += frame0
frames_offset_izq += frame0

while cam_izq.isOpened() or cam_der.isOpened():
    #Se hace el ajuste de los frames desfasados
    while frames_offset_izq>0 or frames_offset_der>0:
        if frames_offset_izq>0:
            cam_izq.read()
            frames_offset_izq -= 1
            frame_counter_izq += 1
        if frames_offset_der>0:
            cam_der.read()
            frames_offset_der -= 1
            frame_counter_der += 1
    
    if show:
        time.sleep(video_delay)

    key_pressed = cv2.waitKey(1)
    if key_pressed == ord('q'):
        break

    elif key_pressed == ord(' '):
        pausa = not pausa
    elif key_pressed == ord('s'):
        save = not save
    elif pausa and key_pressed==ord('i'):
        frame_izq, error_frames = obtener_frame(cam_izq)
        frame_counter_izq += error_frames
    elif pausa and key_pressed==ord('d'):
        frame_der, error_frames = obtener_frame(cam_der)
        frame_counter_der += error_frames
    elif not pausa:
        frame_izq, error_frames = obtener_frame(cam_izq)
        frame_counter_izq += error_frames
        frame_der, error_frames = obtener_frame(cam_der)
        frame_counter_der += error_frames
    
    
    if show:
        img_comp = cv2.hconcat([frame_izq, frame_der])
        img_comp = adjustFrame(img_comp, scale = img_scale)

        cv2.putText(img_comp, frames_to_time(frame_counter_izq),
            org = (50,50),
            fontFace = cv2.FONT_HERSHEY_SIMPLEX,
            fontScale = 1,
            color = (0,0,0)
            )
        cv2.putText(img_comp, frames_to_time(frame_counter_der),
            org = (img_comp.shape[1]-100,50),
            fontFace = cv2.FONT_HERSHEY_SIMPLEX,
            fontScale = 1,
            color = (0,0,0)
            )
        cv2.putText(img_comp, str(abs(frame_counter_der-frame_counter_izq)),
            org = (img_comp.shape[1]//2+10,50),
            fontFace = cv2.FONT_HERSHEY_SIMPLEX,
            fontScale = 1,
            color = (0,0,0)
            )
        cv2.imshow('Frame', img_comp)
    
    if save:
        saveFrame(frame_izq, 'izq/', f'izq_{saved_frames_counter}')
        saveFrame(frame_der, 'der/', f'der_{saved_frames_counter}')
        saved_frames_counter += 1
        if pausa: save = not save

    
cam_izq.release()
cam_der.release()
cv2.destroyAllWindows()