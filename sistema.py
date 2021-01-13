import cv2
import os, sys
import argparse
import time
from glob import glob


from herramientas.general import adjustFrame, obtener_frame
#sys.path.append(os.path.abspath('modelos/yolov5'))
from modelos import localizador, mascaraNCA


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
parser.add_argument('--pesos_loc', type=str,
                    help='Ruta a los pesos del localizador', default='deteccion/pesos/yolo_medium.pt')

args = parser.parse_args()


###Funciones de apoyo ----------------------------------------------------------------------
def dibujar_rois(img, rois, conf_min=0.5):
    for x1, y1, x2, y2, conf in rois:
        if conf>conf_min:
            img = cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0))
            img = cv2.putText(img, f'{conf:.4}', (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),3)
    return img

dir_videos = os.path.abspath(args.data_dir)
p_vid_izq = glob(os.path.join(dir_videos, 'izq*.MP4'))[0]
p_vid_der = glob(os.path.join(dir_videos, 'der*.MP4'))[0]
p_calib_data = os.path.join(dir_videos, 'calib', 'calibData.pk')

video_delay = 0.00
FPS = 25
img_scale = 0.4
frame0 = max(abs(args.offset_d-args.offset_i),
            int(eval(args.frame0)))

cam_izq = cv2.VideoCapture(p_vid_izq)
cam_der = cv2.VideoCapture(p_vid_der)
print('Se inicia el localizador... ', end='')
loc = localizador.Localizador(os.path.abspath(args.pesos_loc))
print('Iniciado :D')
frame_izq, frame_counter_izq = obtener_frame(cam_izq)
frame_der, frame_counter_der = obtener_frame(cam_der)

frames_offset_izq = args.offset_i + frame0
frames_offset_der = args.offset_d + frame0


pausa = True
detectar = False
print('Se inicia la visualización de los videos')
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
        frame_izq = dibujar_rois(frame_izq, regs_izq)
        frame_der = dibujar_rois(frame_der, regs_der)



    time.sleep(video_delay)
    img_comp = cv2.hconcat([frame_izq, frame_der])
    img_comp = adjustFrame(img_comp, scale = img_scale)
    cv2.imshow('Frame', img_comp)

cam_izq.release()
cam_der.release()
cv2.destroyAllWindows()