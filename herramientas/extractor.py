import numpy as np
import cv2
from datetime import date
import os
import time
from glob import glob
from general import frames_to_time, adjustFrame, obtener_frame
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
videos_path = args.data_dir
assert os.path.isdir(videos_path), 'Ingrese un directorio de videos válido'
video_path_izq = glob(os.path.join(videos_path, 'izq*.MP4'))[0]
video_path_der = glob(os.path.join(videos_path, 'der*.MP4'))[0]
assert os.path.isfile(video_path_izq) and os.path.isfile(video_path_der),\
        'Archivos de video no válidos'

dir_to_save = videos_path if args.save_dir=='' else args.save_dir
calib_dir = os.path.join(dir_to_save, 'calib')
imgs_dir = os.path.join(dir_to_save, 'imgs')
video_delay = 0.00
frames_offset_izq = args.offset_i
frames_offset_der = args.offset_d
FPS = 25
img_scale = 0.4
show = True
frame0 = max(abs(frames_offset_der-frames_offset_izq),
            int(eval(args.frame0)))

#---------------------------------------------------------------------------------------

if not os.path.isdir(dir_to_save):
    os.makedirs(dir_to_save)
    os.mkdir(os.path.join(calib_dir,'izq'))
    os.mkdir(os.path.join(calib_dir,'der'))
    os.mkdir(imgs_dir)
    print(f'Se crearon los directorios en {dir_to_save}')


##Si se desea guardar los frames:
def saveFrame(frame, dir=dir_to_save, text=''):
    if not os.path.isdir(dir):
        os.makedirs(dir)
    name = os.path.join(dir,f'frame_{text}.png')
    cv2.imwrite(name, frame)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((pattern_size[0]*pattern_size[1],3), np.float32)
objp[:,:2] = np.mgrid[0:pattern_size[0],0:pattern_size[1]].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


## Comienza la captura del video
cam_izq = cv2.VideoCapture(video_path_izq)
cam_der = cv2.VideoCapture(video_path_der)
#error_frames_counter = 0
saved_frames_counter = 1
#frame_counter = 0
frame_counter_der = 0
frame_counter_izq = 0
pausa = True
save = False

frame_izq, frame_counter_izq = obtener_frame(cam_izq)
frame_der, frame_counter_der = obtener_frame(cam_der)

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
    elif key_pressed == ord('c'):
        new_frame_izq = frame_izq.copy()
        new_frame_der = frame_der.copy()
        gray_izq = cv2.cvtColor(frame_izq, cv2.COLOR_BGR2GRAY)
        gray_der = cv2.cvtColor(frame_der, cv2.COLOR_BGR2GRAY)
        ret_izq, corners_izq = cv2.findChessboardCorners(gray_izq, pattern_size, 
                                            flags = cv2.CALIB_CB_NORMALIZE_IMAGE+\
                                                    cv2.CALIB_CB_ADAPTIVE_THRESH+\
                                                    cv2.CALIB_CB_FILTER_QUADS)
        
        ret_der, corners_der = cv2.findChessboardCorners(gray_der, pattern_size, 
                                            flags = cv2.CALIB_CB_NORMALIZE_IMAGE+\
                                                    cv2.CALIB_CB_ADAPTIVE_THRESH+\
                                                    cv2.CALIB_CB_FILTER_QUADS)
    
        if ret_izq and ret_der:
            temp_izq = frame_izq.copy()
            temp_der = frame_der.copy()
            cv2.drawChessboardCorners(temp_izq, pattern_size, corners_izq, ret_izq)
            cv2.drawChessboardCorners(temp_der, pattern_size, corners_der, ret_der)
            temp_comp = cv2.hconcat([temp_izq, temp_der])
            temp_comp = adjustFrame(temp_comp, scale = img_scale)
            cv2.imshow('Frame', temp_comp)
        else:
            cv2.imshow('Frame', np.zeros(img_comp.shape[:2]))

        desicion_key = cv2.waitKey(0)
        if desicion_key == ord('s'): #and ret_der and ret_izq:
            saveFrame(new_frame_izq, os.path.join(calib_dir, 'izq'),
                     f'izq_{saved_frames_counter}')
            saveFrame(new_frame_der, os.path.join(calib_dir, 'der'),
                     f'der_{saved_frames_counter}')
            saved_frames_counter += 1
        elif desicion_key == ord(' '):
            pausa = False

    elif not pausa:
        frame_izq, error_frames = obtener_frame(cam_izq)
        frame_counter_izq += error_frames
        frame_der, error_frames = obtener_frame(cam_der)
        frame_counter_der += error_frames



    time.sleep(video_delay)
    
    img_comp = cv2.hconcat([frame_izq, frame_der])
    img_comp = adjustFrame(img_comp, scale = img_scale)

    cv2.putText(img_comp, frames_to_time(frame_counter_izq, FPS),
            org = (50,50),
            fontFace = cv2.FONT_HERSHEY_SIMPLEX,
            fontScale = 1,
            color = (0,0,0)
            )
    cv2.putText(img_comp, frames_to_time(frame_counter_der, FPS),
        org = (img_comp.shape[1]-100,50),
        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
        fontScale = 1,
        color = (0,0,0)
        )
    cv2.putText(img_comp, str(abs(frame_counter_der-frame_counter_izq)),
        org = (img_comp.shape[1]//2+10,50),
        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
        fontScale = 1,
        color = (0,0,255)
        )
    cv2.imshow('Frame', img_comp)

    if save:
        saveFrame(frame_izq, os.path.join(imgs_dir, 'izq'),
                 f'izq_{saved_frames_counter}')
        saveFrame(frame_der, os.path.join(imgs_dir, 'der'),
                 f'der_{saved_frames_counter}')
        saved_frames_counter += 1
        if pausa: save = not save
    
cam_izq.release()
cam_der.release()
cv2.destroyAllWindows()