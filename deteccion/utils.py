import matplotlib
import matplotlib.pyplot as plt
import glob
import cv2
from PIL import Image, ImageDraw, ImageFont
import os
import sys
import numpy as np
from scipy.spatial.distance import euclidean
from skimage.filters import threshold_minimum, gaussian, threshold_triangle
from skimage.morphology import skeletonize
from skimage.exposure import rescale_intensity
from skimage.feature import match_descriptors, ORB, plot_matches
from skimage.color import rgb2gray


class LectorVideo():
    def __init__(self, video_path, offset= 0, FPS = 25):
        assert os.path.exists(video_path)
        self.FPS = FPS

        self.video = cv2.VideoCapture(video_path)

        self.frame_counter = -1

        while self.video.isOpened() and offset>0:
            self.video.read()
            offset -= 1

    def siguiente(self, max_frames_error=50):
        while self.video.isOpened():
            ret, frame = self.video.read()

            if not ret and 0<max_frames_error:
                max_frames_error -= 1
                continue

            self.frame_counter += 1

            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None


def dibujar_rois(img, res, linewidth = 2):
    """
    Entrada:
    	img -> numpy array
    	res -> diccionario de la detección
    Salida:
    	Objeto de Imagen PIL con las rois dibujadas
    """

    colors = ['DarkViolet','Crimson', 'Chartreuse', 'Cyan', 'DarkOrange', 
                'DeepPink', 'MediumBlue', 'Coral', 'DarkRed', 'DeepSkyBlue',
                'DodgerBlue', 'Fuchsia', 'Indigo', 'LawnGreen', 'Lime',
                'Magenta', 'MediumBlue', 'MediumVioletRed', 'Navy', 'OrangeRed',
                'Red', 'Purple', 'Yellow']

    rois = res['rois']
    scores = res['scores']

    pil_img = Image.fromarray(img)
    canvas = ImageDraw.Draw(pil_img)

    if len(rois)>len(colors):
        colors = colors + colors

    for roi, score, color in zip(rois, scores, colors):
        y1, x1, y2, x2  = roi
        canvas.rectangle([x1, y1, x2, y2], outline = color, width = linewidth)
        canvas.text((x1,y1), f'{score}', fill = color)

    return pil_img

def extraer_rois(img, rois):
    """
    Entrada:
        img -> numpy array
        rois -> numpy array con un array por cada roi a extraer
    Salida
        ista con cada región extraída
    """
    regiones = []

    for roi in rois:
        y1, x1, y2, x2  = roi
        regiones.append(img[y1:y2, x1:x2])

    return regiones


def obtener_region_punto(punto, shape, t_ventana=20):
    """
    Entrada
        punto - Coordenadas bidimensionales de un punto dentro de la imagen
        shape - tupla (alto, ancho) de las dimensiones de la imagen
        t_ventana - tamaño de la ventana de la región (cuadrada)
    Salida
        Bouning box (y0,y1,x0,x1) de la region

    El algoritmo contempla que la región deseada puede caer fuera de la imagen,
    por lo que se hacen los ajustes necesarios para que siempre se encuentre
    dentro de la imagen.
    """
    def ajustar_coordenadas(val, minimo, maximo, t_ventana):
        val0 = val-t_ventana//2
        val1 = val+t_ventana//2
        if val0<minimo:
            val0 = minimo
            val1 = t_ventana
        elif maximo-1<val1:
            val1 = maximo-1
            val0 = val1-t_ventana
        return val0, val1
    alto, ancho = shape[:2]
    y, x = punto
    y0, y1 = ajustar_coordenadas(y, 0, alto, t_ventana)
    x0, x1 = ajustar_coordenadas(x, 0, ancho, t_ventana)

    return (y0, y1, x0, x1)




def obtener_puntos(img, sigma = 2.5, devolver_imagenes = False):
    """
    Entrada
    img -> imagen del pez de la que se desean conocer los puntos característicos
            del esqueleto
    sigma -> Sigma para el filtro gaussiano
    devolver_imagenes -> Si es True, devuelve un diccionario con las imágenes
                        procesadas: mask (la máscara binaria del pez), esqueleto
                        (resultado de la esqueletización de la máscara) y
                        proyeccion (imagen con los puntos proyectados)
    Salida
    tupla (puntos_finales, intersecciones, *imagenes*)
    """

    color_adjusted_img = rescale_intensity(img)
    smothed = gaussian(rgb2gray(color_adjusted_img), sigma)
    threshold = threshold_triangle(smothed)
    mask = smothed<threshold
    esqueleto = skeletonize(mask)

    #Función para buscar los puntos finales e intersecciones
    distancia_minima = 5
    candidatos_filas, candidatos_columnas = np.nonzero(esqueleto)
    max_filas, max_columnas = esqueleto.shape
    puntos_finales = []
    intersecciones = []
    for fil, col in zip(candidatos_filas, candidatos_columnas):
        if fil == 0 or col == 0 or fil==max_filas-1 or col==max_columnas-1:
            puntos_finales.append(np.array([fil,col]))
            continue
        n_conexiones = esqueleto[fil-1:fil+2, col-1:col+2].sum()
        if n_conexiones == 2:
            for u in puntos_finales:
                distancia = euclidean(np.array(u), np.array([fil,col]))
                if distancia<distancia_minima:
                    break
            else:
                puntos_finales.append(np.array([fil,col]))

        elif n_conexiones == 4:
            for u in intersecciones:
                distancia = euclidean(np.array(u), np.array([fil,col]))
                if distancia<distancia_minima:
                    break
            else:
                intersecciones.append(np.array([fil,col]))

    if devolver_imagenes:
        img_canvas = Image.fromarray(color_adjusted_img)
        canvas = ImageDraw.Draw(img_canvas)
        for y, x in puntos_finales:
            canvas.ellipse([(x-1,y-1),(x+2, y+2)], fill='red')
        for y, x in intersecciones:
            canvas.ellipse([(x-1,y-1),(x+2, y+2)], fill='blue')
        imagenes = {'mask': mask,
                    'esqueleto': esqueleto,
                    'proyeccion': img_canvas}
        
        return puntos_finales, intersecciones, imagenes
    else:
        return puntos_finales, intersecciones


def emparejar_rois(frame1, frame2, deteccion1, deteccion2, max_ratio=0.8, plot=False):
    """
    Entrada
        frame1, frame2 -> Cuadros extraídos de ambas cámaras
        deteccion1, deteccion2 -> Resultado de la detección generada por la RNA
        max_ratio -> Argumento de la función Match_Descriptors
        plot -> Si es True, graficará el emparejamiento obtenido
    Salida
        array de tamaño mx2, donde cada fila contiene las dos rois de las regiones
            emparejadas
    """
    invertir=False
    regiones1 = [rescale_intensity(img) for img in extraer_rois(frame1, deteccion1['rois'])]
    regiones2 = [rescale_intensity(img) for img in extraer_rois(frame2, deteccion2['rois'])]

    descriptor = ORB(n_keypoints=500, harris_k=0.04,
                    downscale = 1.2, n_scales = 10)
    desc_reg1 = []
    for reg in regiones1:
        descriptor.detect_and_extract(rgb2gray(reg))
        desc_reg1.append([descriptor.keypoints,
                        descriptor.descriptors])

    desc_reg2 = []
    for reg in regiones2:
        descriptor.detect_and_extract(rgb2gray(reg))
        desc_reg2.append([descriptor.keypoints,
                        descriptor.descriptors])

    
    scores = np.zeros((len(regiones1), len(regiones2)))
    for i, (kp1, desc1) in enumerate(desc_reg1):
        for j, (kp2, desc2) in enumerate(desc_reg2):
            matches = match_descriptors(desc1, desc2, cross_check=True,
                                        max_ratio=max_ratio)
            puntaje = len(matches)/len(desc1)*100
            scores[i, j] = puntaje
    
    if np.argmin(scores.shape)==1:
        invertir=True
        scores = scores.T
        
    mejores = [np.argmax(x) for x in scores]
    rois_emparejadas=[]
    for primera, segunda in enumerate(mejores):
        if invertir:
            rois_emparejadas.append((deteccion1['rois'][segunda],
                                    deteccion2['rois'][primera]))
        else:
            rois_emparejadas.append((deteccion1['rois'][primera],
                                    deteccion2['rois'][segunda]))
        if plot:
            fig, ax = plt.subplots(figsize=(10,5))
            if not invertir:
                kp1, desc1 =desc_reg1[primera]
                kp2, desc2 =desc_reg2[segunda]
                matches = match_descriptors(desc1, desc2, cross_check=True,
                                                max_ratio=max_ratio)
                plot_matches(ax, regiones1[primera], regiones2[segunda], kp1, kp2, matches)
            else:
                kp1, desc1 =desc_reg1[segunda]
                kp2, desc2 =desc_reg2[primera]
                matches = match_descriptors(desc1, desc2, cross_check=True,
                                                max_ratio=max_ratio)
                plot_matches(ax, regiones1[segunda], regiones2[primera], kp1, kp2, matches)
            plt.show()
    
    return rois_emparejadas

def determinar_cabeza_aleta(puntos, intersecciones):
    """
    Entrada:
        puntos-> Lista de arrays con los puntos finales del esqueleto del pez
        intersecciones -> Lista de arrays con las intersecciones del esqueleto
    Salida:
        tupla de arrays con las coordenadas (Cabeza, Aleta)
    """
    def angulo_en_tres_puntos(p1, p2, p3):
        p12 = p1-p2
        p23 = p1-p3
        cosine_angle = np.dot(p12, p23) / (np.linalg.norm(p12) * np.linalg.norm(p23))
        return np.arccos(cosine_angle)
    
    puntos = sorted(puntos, key = lambda x: x[1])
    intersecciones = sorted(intersecciones, key = lambda x: x[1])
    p1 = puntos[0]
    p2 = puntos[-1]
    int1 = intersecciones[0]
    int2 = intersecciones[-1]
    pendiente_izq = angulo_en_tres_puntos(p1, int1, int2)
    pendiente_der = angulo_en_tres_puntos(int1, int2, p2)

    if pendiente_izq<=pendiente_der:
        #print('cabeza del lado izquierdo')
        return p1, int2
    else:
        #print('cabeza del lado derecho')
        return p2, int1


if __name__=='__main__':
    a = np.array([[1,2,3],
                  [1,2,3],
                  [4,5,6],
                  [3,4,5]])
    print(a.shape)
    print(np.argmin(a.shape))