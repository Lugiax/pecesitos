import os
import numpy as np
import cv2 as cv
from glob import glob
import csv

def obtener_id(path):
    return os.path.basename(path).split('.')[0]

def obtener_datos(anot_path):
    img_id = obtener_id(anot_path)
    img = cv.imread(os.path.join(imgs_dir, img_id+'.jpg'))
    regiones = []
    with open(anot_path, 'r') as f:
        reader = csv.reader(f)
        for r in reader:
            x1,x2,y1,y2 = [int(v) for v in r[:4]]
            H,W= region_size
            p1y,p1x,p2y,p2x = [float(v) for v in r[4:]]
            p1y, p2y = int(p1y*H), int(p2y*H)
            p1x, p2x = int(p1x*W), int(p2x*W)
            regiones.append((img[y1:y2, x1:x2],
                            ([p1y,p1x],[p2y,p2x])))
    return regiones



def dibujar_puntos(img, puntos):
    tipo = ['Boca', 'Aleta']
    for i, p in enumerate(puntos):
        if p is not None:
            cv.putText(img, tipo[i], tuple(p[::-1]), cv.FONT_HERSHEY_SIMPLEX,
                       0.7, (250*i, 0, 250*abs(i-1)), 1)
            cv.circle(img, tuple(p[::-1]), 2, (250*i, 0, 250*abs(i-1)), -1)


data_dir = '/home/carlos/Documentos/Codes/ProyectoMCC/datos/'
anotaciones_dir = os.path.join(data_dir, 'PosicionesBocaAleta')
imgs_dir = os.path.join(data_dir, 'imagenes')
anot_list = glob(os.path.join(anotaciones_dir, '*.txt'))
np.random.shuffle(anot_list)

region_size = (350,600)
nuevo = True


cv.namedWindow('imagen')
while True:
    if nuevo:
        puntos_guardados = []
        anot_path = anot_list.pop()
        assert anot_path is not None, 'Ya no hay nuevas máscaras a revisar'
        regiones = obtener_datos(anot_path)
        region, puntos = regiones.pop()
        nuevo=False

    a_mostrar = cv.resize(region, region_size[::-1])
    dibujar_puntos(a_mostrar, puntos)
    cv.imshow('imagen', a_mostrar)

    key = cv.waitKey(50)
    if key==ord('q'):
        break
    elif key==ord('n'):
        if len(regiones)>0:
            region, puntos=regiones.pop()
        elif len(regiones)==0:
            nuevo = True
        


cv.destroyAllWindows()