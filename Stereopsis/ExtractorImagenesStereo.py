import numpy as np
import cv2
from datetime import date
import os
import time
from utils import adjustFrame

pattern_size = (8,6)
video_path_izq = '/home/carlos/Documentos/Codes/ProyectoMCC/Stereopsis/imgs/videocal/pruebas_robalo/toma1_izq.MP4'
video_path_der = '/home/carlos/Documentos/Codes/ProyectoMCC/Stereopsis/imgs/videocal/pruebas_robalo/toma1_der.MP4'
dir_to_save = '/home/carlos/Documentos/Codes/ProyectoMCC/Stereopsis/imgs/videocal/pruebas_robalo/imgs_toma1_calib'
video_delay = 0.00
frames_offset_izq = (60+20)*25+3
frames_offset_der = (60+18)*25
FPS = 25
img_scale = 0.4

#---------------------------------------------------------------------------------------

assert os.path.exists(video_path_izq)
assert os.path.exists(video_path_der)

if not os.path.isdir(dir_to_save):
    os.makedirs(os.path.join(dir_to_save,'izq'))
    os.makedirs(os.path.join(dir_to_save,'der'))
    print(f'Se creó el directorio {dir_to_save}')


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

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((pattern_size[0]*pattern_size[1],3), np.float32)
objp[:,:2] = np.mgrid[0:pattern_size[0],0:pattern_size[1]].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


## Comienza la captura del video
cam_izq = cv2.VideoCapture(video_path_izq)
cam_der = cv2.VideoCapture(video_path_der)
error_frames_counter = 0
saved_frames_counter = 0
frame_counter = 0
while cam_izq.isOpened():
    #Se hace el ajuste de los frames desfasados
    while frames_offset_izq>0 or frames_offset_der>0:
        if frames_offset_izq>0:
            cam_izq.read()
            frames_offset_izq -= 1
        if frames_offset_der>0:
            cam_der.read()
            frames_offset_der -= 1
        

    time.sleep(video_delay)

    ret_izq, frame_izq = cam_izq.read()
    ret_der, frame_der = cam_der.read()
    ##Sección para evitar errores en la captura de frames
    max_frames_error = 50
    if (not ret_izq or not ret_der) and error_frames_counter<max_frames_error: 
        error_frames_counter+=1
        frame_counter+=1
        continue
    elif error_frames_counter==max_frames_error:
        print('LÍMITE DE FRAMES CON ERROR ALCANZADO. SALIENDO.')
        break
    error_frames_counter = 0
    
    img_comp = cv2.hconcat([frame_izq, frame_der])
    img_comp = adjustFrame(img_comp, scale = img_scale)

    cv2.putText(img_comp, frames_to_time(frame_counter),
        org = (50,50),
        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
        fontScale = 1,
        color = (0,0,0)
        )
    cv2.imshow('Frame', img_comp)
    
    key_pressed = cv2.waitKey(1)
    if key_pressed == ord('q'):
        break
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
            
            cv2.drawChessboardCorners(frame_izq, pattern_size, corners_izq, ret_izq)
            cv2.drawChessboardCorners(frame_der, pattern_size, corners_der, ret_der)
            img_comp = cv2.hconcat([frame_izq, frame_der])
            img_comp = adjustFrame(img_comp, scale = img_scale)
            cv2.imshow('Frame', img_comp)
        #else:
        #    cv2.imshow('Frame', np.zeros(img_comp.shape[:2]))

        desicion_key = cv2.waitKey(0)
        if desicion_key == ord('s'): #and ret_der and ret_izq:
            saveFrame(new_frame_izq, 'izq/', f'izq_{saved_frames_counter}')
            saveFrame(new_frame_der, 'der/', f'der_{saved_frames_counter}')
            saved_frames_counter += 1

    frame_counter += 1
    
cam_izq.release()
cam_der.release()
cv2.destroyAllWindows()