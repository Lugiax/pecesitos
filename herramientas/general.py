#%utf8
import cv2
import os
from glob import glob
from skimage.feature import match_descriptors, ORB, plot_matches
from skimage.color import rgb2gray
from skimage.exposure import adjust_sigmoid, match_histograms
from skimage.filters import gaussian
import numpy as np

def procesar_Sisal(img, otra_img=None):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = adjust_sigmoid(rgb, cutoff=0.5, gain=10, inv=False)
    rgb = gaussian(rgb, sigma = 3, multichannel=True)
    if otra_img is not None:
        ref = cv2.cvtColor(otra_img, cv2.COLOR_BGR2RGB)
        rgb = match_histograms(rgb, ref, multichannel=True)
    return cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2BGR)


def obtener_frame(camara, frame=None, ret=False, voltear=False, proc_func=None, otra_img=None):
    max_frames_error = 50
    error_frames_counter = 0
    while not ret and camara.isOpened():
        if not ret and error_frames_counter<max_frames_error: 
            error_frames_counter+=1
            ret, frame = camara.read()
        elif error_frames_counter==max_frames_error:
            print('LIMITE DE FRAMES CON ERROR ALCANZADO. SALIENDO.')
            camara.release()
            break
    if voltear:
        frame = cv2.rotate(frame, cv2.ROTATE_180)

    if proc_func is not None:
        frame = proc_func(frame, otra_img)

    return frame, error_frames_counter

def adjustFrame(frame, scale = 1):
    H, W = frame.shape[:2]
    return cv2.resize(frame, (int(W*scale), int(H*scale)),
                        interpolation = cv2.INTER_CUBIC)

def frames_to_time(frames, FPS=25):
    segundos = frames//FPS
    tiempo = [str(segundos)]
    frames %= FPS
    if segundos>59:
        minutos = segundos//60
        segundos %= 60
        tiempo = ['{:0>2}'.format(str(minutos)),
                '{:0>2}'.format(str(segundos))]
    return(':'.join(tiempo)+'.{:0>2}'.format(frames))

def extraer_rois(img, rois):
	"""
	Entrada:
		img -> numpy array
		rois -> numpy array con un array por cada roi a extraer
	Salida
		Lista con cada región extraída
	"""
	regiones = []
	for roi in rois:
		x1, y1, x2, y2  = roi[:4]
		regiones.append(img[y1:y2, x1:x2])

	return regiones

def emparejar_rois(frame1, frame2, rois1, rois2, plot=False):
    """
    Entrada
        frame1, frame2 -> Cuadros extraídos de ambas cámaras
        deteccion1, deteccion2 -> Resultado de la detección generada por la RNA
        plot -> Si es True, graficará el emparejamiento obtenido
    Salida
        array de tamaño mx2, donde cada fila contiene las dos rois de las regiones
            emparejadas
    """
    regiones1 = extraer_rois(frame1, rois1)#[rescale_intensity(img) for img in extraer_rois(frame1, deteccion1['rois'])]
    regiones2 = extraer_rois(frame2, rois2)#[rescale_intensity(img) for img in extraer_rois(frame2, deteccion2['rois'])]

    descriptor = ORB(n_keypoints=1000, harris_k=0.01,
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
                                        max_ratio=0.8)
            puntaje = len(matches)/len(desc1)*100
            scores[i, j] = puntaje

    mejores = [np.argmax(x) for x in scores]
    rois_emparejadas=[]
    for img1, img2 in enumerate(mejores):
        rois_emparejadas.append((rois1[img1],
                                 rois2[img2]))
        if plot:
            fig, ax = plt.subplots(figsize=(10,5))
            kp1, desc1 =desc_reg1[img1]
            kp2, desc2 =desc_reg2[img2]
            matches = match_descriptors(desc1, desc2, cross_check=True,
                                            max_ratio=0.8)
            plot_matches(ax, regiones1[img1], regiones2[img2], kp1, kp2, matches)
            plt.show()
    
    return rois_emparejadas


class Grabador:
    def __init__(self, nombre='grabacion.MP4', fps=25):
        self.nombre = nombre
        self.fps = fps
        self.escritor = None
    
    def desde_archivo(self, ruta, offset=100, max_frames=100):
        assert os.path.isfile(ruta), 'Verificar ruta del archivo'
        cam = cv2.VideoCapture(ruta)

        if offset>0: print('Recorriendo video...')
        while offset>0:
            cam.read()
            offset -= 1
        sig, _ = obtener_frame(cam)
        n_frames = 1
        #print(n_frames, cam.isOpened(), sig.shape)
        while cam.isOpened():
            self.agregar(sig)
            if max_frames>0 and n_frames >= max_frames:
                break
            sig, _ = obtener_frame(cam)
            n_frames += 1
            #print(n_frames, cam.isOpened(), sig.shape)
        self.escritor.release()
    
    def desde_carpeta(self, carpeta):
        formatos = ['jpg', 'png']
        imgs = sorted([f for f in glob(os.path.join(carpeta,'*.*')) if f.split('.')[1] in formatos])
        for img_p in imgs:
            self.agregar(cv2.imread(img_p))
        self.escritor.release()


    def agregar(self, frame):
        #agregar un frame a la secuencia
        if self.escritor is None:
            print('Creando el escritor')
            self.escritor = cv2.VideoWriter(self.nombre, cv2.VideoWriter_fourcc(*'mp4v'),
                                            self.fps, tuple(frame.shape[:2][::-1]))
        
        self.escritor.write(frame)
    
    def terminar(self):
        self.escritor.release()


if __name__=='__main__':
    from skimage.io import imread
    from skimage.exposure import adjust_log, rescale_intensity, adjust_sigmoid
    import matplotlib.pyplot as plt
    import cv2
    img_path = '/home/carlos/Vídeos/Sisal/2/res_sis/imgs/der/frame_der_9.png'
    img = cv2.imread(img_path)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1,3, figsize=(15,5))
    rgb1 = rescale_intensity(adjust_sigmoid(rgb, cutoff=0.5, gain=10, inv=False))#adjust_gamma(adjust_sigmoid(rgb, cutoff=0.5, gain=10, inv=False), gamma=0.7)
    ax[0].imshow(rgb1)
    rgb1 = rescale_intensity(adjust_sigmoid(rgb, cutoff=0.5, gain=10, inv=False))
    ax[1].imshow(rgb1)
    rgb1 = rescale_intensity(adjust_sigmoid(rgb, cutoff=0.5, gain=10, inv=False))
    ax[2].imshow(rgb1)
    plt.show()




