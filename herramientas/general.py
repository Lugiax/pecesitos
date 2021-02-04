import cv2
import os

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

class Grabador:
    def __init__(self, nombre='grabacion.mp4', fps=25):
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
    g = Grabador('/media/carlos/Archivos/PROYECTO/angulos2/1/grabacion.mp4')
    g.desde_archivo(('/media/carlos/Archivos/PROYECTO/angulos2/1/izq1.MP4'))

