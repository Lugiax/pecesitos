import os
import numpy as np
import cv2 as cv
import pickle
from scipy.spatial.distance import euclidean

directorio = '/mnt/1950EF8830BF5793/PROYECTO/angulos/180g'
frame = 51
path_img_i = os.path.join(directorio, f'imgs/izq/frame_izq_{frame}.png')
path_img_d = os.path.join(directorio, f'imgs/der/frame_der_{frame}.png')
calib_data_path = os.path.join(directorio, 'calib/calibData.pk')

factor_escala = 2
ventana = (100,100)

###-------------------------------------------------------------------------
def mouse(evento, x, y, flags, param):
    global mouse_x, mouse_y, raton_en_pantalla
    global primer_punto, puntos, puntos_agregados
    mouse_x = x
    mouse_y = y
    raton_en_pantalla = True

    if evento==cv.EVENT_LBUTTONDOWN:
        if not primer_punto:
            puntos.append([y,x])
            primer_punto = True
        elif primer_punto:
            #Se acomodan según la imagen que corresponde, ordenándolos según x
            puntos.append([y,x])
            puntos = sorted(puntos, key=lambda x: x[1])
            puntos_agregados = True
            primer_punto = False

def roi_interna(p, ventana, shape):
    x, y = p
    v_height, v_width = ventana
    height, width = shape

    x0 = max(0, x-v_width//2) if x<width-v_width//2 else width-v_width
    x1 = min(width, x+v_width//2) if x>v_width//2 else v_width
    y0 = max(0, y-v_height//2) if y<height-v_height//2 else height-v_height
    y1 = min(height, y+v_height//2) if y>v_height//2 else v_height

    return(x0,y0,x1,y1)

def dibujar_puntos(img, pts1, pts2, factor_escala=factor_escala, x_offset=0,
                    size=2, color = (0, 0, 250)):
    pts1 = [[int(pi/factor_escala) for pi in pt] for pt in pts1]
    pts2 = [[int(pt[0]/factor_escala),
                     int(pt[1]/factor_escala)+x_offset] for pt in pts2]

    for p1, p2 in zip(pts1, pts2):
        cv.circle(img, tuple(p1[::-1]), size, color, -1)
        cv.circle(img, tuple(p2[::-1]), size, color, -1)

##------------------------------------------------------------------------------

img1_o = cv.imread(path_img_i)
img2_o = cv.imread(path_img_d)

assert img1_o.shape==img2_o.shape, 'Las imágenes no tienen las mismas dimensiones'

with open(calib_data_path, 'rb') as f:
    stereo_params = pickle.load(f)

##Se hace el cálculo de la rectificación
R1, R2, P1, P2, Q, _, _ = cv.stereoRectify(stereo_params['cameraMatrix1'],
                                          stereo_params['distCoeffs1'],
                                          stereo_params['cameraMatrix2'],
                                          stereo_params['distCoeffs2'],
                                          img1_o.shape[:2][::-1],
                                          stereo_params['R'],
                                          stereo_params['T'],
                                          flags=cv.CALIB_ZERO_DISPARITY)


img = cv.hconcat([img1_o, img2_o])
resized_shape = tuple([int(n/factor_escala) for n in img.shape[:2]][::-1])
print('Resized concat shape=', resized_shape)
img = cv.resize(img, resized_shape)
ancho_ventana = img.shape[1]


mouse_x, mouse_y= 0,0
raton_en_pantalla = False

puntos_i = []
puntos_d = []
puntos_corr_i = []
puntos_corr_d = []
puntos3D = []
puntos = []
primer_punto = False
puntos_agregados = False
tamano_cuadro = stereo_params.get('tamano_cuadro', 25)
unidades='mm'

nombre_ventana = f'cuadro {frame}'
cv.namedWindow(nombre_ventana)
cv.setMouseCallback(nombre_ventana,mouse)
while True:
    temporal = img.copy()

    dibujar_puntos(temporal, puntos_i, puntos_d, x_offset=ancho_ventana//2)
    dibujar_puntos(temporal, puntos_corr_i, puntos_corr_d,
                    x_offset=ancho_ventana//2, color=(255,0,0))

    if raton_en_pantalla:
        #print('Pos mouse:',mouse_x, mouse_y)
        #print('Ancho ventanta:', ancho_ventana)
        roi = roi_interna((mouse_x, mouse_y), ventana, temporal.shape[:2])
        x_centro, y_centro = None, None
        if mouse_x<ancho_ventana//2-ventana[1]//2-1:
            x_centro, y_centro = map(lambda x: int(x*factor_escala), (mouse_x, mouse_y))
            roi_o= roi_interna((x_centro, y_centro), ventana, img1_o.shape[:2])
            temporal[roi[1]:roi[3], roi[0]:roi[2]] = img1_o[roi_o[1]:roi_o[3],
                                                            roi_o[0]:roi_o[2]]
        elif mouse_x>=ancho_ventana//2+ventana[1]//2+2:
            x_centro, y_centro = map(lambda x: int(x*factor_escala),
                                        (mouse_x-ancho_ventana//2, mouse_y))
            roi_o= roi_interna((x_centro, y_centro), ventana, img1_o.shape[:2])
            temporal[roi[1]:roi[3], roi[0]:roi[2]] = img2_o[roi_o[1]:roi_o[3],
                                                            roi_o[0]:roi_o[2]] 
        else:
            raton_en_pantalla = False

    if puntos_agregados:
        p1, p2 = puntos
        if p1[1]<ancho_ventana//2 and p2[1]>=ancho_ventana//2:
            p1 = [int(pi*factor_escala) for pi in p1]
            p2 = [int(p2[0]*factor_escala),
                             int((p2[1]-ancho_ventana//2)*factor_escala)]
            p1_corr = np.array([[p1[::-1]]], dtype=np.float32)
            p2_corr = np.array([[p2[::-1]]], dtype=np.float32)
            corregidos = cv.correctMatches(stereo_params['F'], p1_corr, p2_corr)
            p1_corr = corregidos[0][0]
            p2_corr = corregidos[1][0]
            #print(p1_corr.T.shape, p1_corr.T)
            triangulacion = cv.convertPointsFromHomogeneous(cv.triangulatePoints(P1,
                                                                                 P2,
                                                                                 p1_corr.T,
                                                                                 p2_corr.T).T
                                                            )
            puntos3D.insert(0, triangulacion[0][0])

            puntos_i.insert(0, p1)
            puntos_d.insert(0, p2)
            puntos_corr_i.insert(0, p1_corr[0][::-1])
            puntos_corr_d.insert(0, p2_corr[0][::-1])
            print(f'Se agregan los puntos {puntos_i[-1]} y {puntos_d[-1]};'
                  f' corregidos {puntos_corr_i[-1]} y {puntos_corr_d[-1]};'
                  f' 3D {puntos3D[-1]}')
            if len(puntos3D)>2:
                puntos3D.pop();puntos_i.pop();puntos_d.pop()
                puntos_corr_i.pop();puntos_corr_d.pop()
            if len(puntos3D)==2:
                distancia = euclidean(*puntos3D)*tamano_cuadro
                for p in puntos3D:
                    print(f'La distancia al punto {p} es de {np.linalg.norm(p)*tamano_cuadro}{unidades}')
                print(f'Distancia entre los puntos {puntos3D[0]} y {puntos3D[1]} es de '
                      f'{distancia}{unidades}')

        puntos=[]
        puntos_agregados=False

    cv.imshow(nombre_ventana, temporal)

    key = cv.waitKey(50)
    if key==ord('q'):
        break

cv.destroyAllWindows()