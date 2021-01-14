import cv2

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