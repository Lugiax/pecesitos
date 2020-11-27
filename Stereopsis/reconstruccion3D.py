import os
import numpy as np
import cv2 as cv

path_img_i = 'imgs/videocal/pruebas_robalo/Montaje/IMG_20201104_113610.jpg'
path_img_d = 'imgs/videocal/pruebas_robalo/Montaje/IMG_20201104_113617.jpg'

factor_escala = 3
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



img1_o = cv.imread(path_img_i)
img2_o = cv.imread(path_img_d)
print('Imgs shape:', img1_o.shape)


img = cv.hconcat([img1_o, img2_o])
resized_shape = tuple([int(n/factor_escala) for n in img.shape[:2]][::-1])
print('Resized concat shape=', resized_shape)
img = cv.resize(img, resized_shape)
ancho_ventana = img.shape[1]


mouse_x, mouse_y= 0,0
raton_en_pantalla = False

puntos_i = []
puntos_d = []
puntos = []
primer_punto = False
puntos_agregados = False

cv.namedWindow('imagen')
cv.setMouseCallback('imagen',mouse)
while True:
    temporal = img.copy()
    if raton_en_pantalla:
        print('Pos mouse:',mouse_x, mouse_y)
        print('Ancho ventanta:', ancho_ventana)
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
            puntos_i.append([int(pi*factor_escala) for pi in p1])
            puntos_d.append([int(p2[0]*factor_escala),
                             int((p2[1]-ancho_ventana//2)*factor_escala)])
            cv.circle(img, tuple(p1[::-1]), 5, (0, 0, 250), -1)
            cv.circle(img, tuple(p2[::-1]), 5, (0, 0, 250), -1)
            print(f'Se agregan los puntos {puntos_i[-1]} y {puntos_d[-1]}')
        
        puntos=[]
        puntos_agregados=False

    cv.imshow('imagen', temporal)

    key = cv.waitKey(50)
    if key==ord('q'):
        break

cv.destroyAllWindows()