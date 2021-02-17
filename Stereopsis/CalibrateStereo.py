import numpy as np
import cv2
from datetime import date
import os
import time
from glob import glob
import pickle
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str,
                    help='Directorio base donde se encuentran las imágenes para calibrar')
parser.add_argument('tamano_cuadro', type=int,
                    help='Tamaño de un cuadro del tablero de calibración')
parser.add_argument('--save_dir', type=str,
                    help='Directorio relativo al directorio de datos donde se guardarán los resultados',
                    default='')
parser.add_argument('--error', type=float,
                    help='Error promedio máximo para la calibración, en píxeles',
                    default=1)
parser.add_argument('--factor', type=float,
                    help='Factor de disminución del error',
                    default=1)
parser.add_argument('--imgs_min', type=int,
                    help='Número mínimo de imágenes para hacer la calibración',
                    default = 15)
parser.add_argument('--manual', type=bool, default=False)

args = parser.parse_args()

pattern_size = (8,6)
imgs_dir = os.path.abspath(args.data_dir)
imgs_izq_dir = os.path.join(imgs_dir, 'izq')
imgs_der_dir = os.path.join(imgs_dir, 'der')
save_dir = os.path.join(imgs_dir, args.save_dir)
error_promedio_min = args.error #pixeles
factor_de_disminucion = args.factor
n_imgs_min = args.imgs_min

assert os.path.exists(imgs_izq_dir) and os.path.exists(imgs_der_dir),\
        'Los directorios de las imágenes no existen'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f'Se ha creado el directorion {save_dir}')
#---------------------------------------------------------------------------------------
def buscar_y_extraer(img1, img2):
    ret1, corners1 = cv2.findChessboardCorners(img1, pattern_size, 
                                        flags = cv2.CALIB_CB_NORMALIZE_IMAGE+\
                                                cv2.CALIB_CB_ADAPTIVE_THRESH+\
                                                cv2.CALIB_CB_FILTER_QUADS)

    ret2, corners2 = cv2.findChessboardCorners(img2, pattern_size, 
                                        flags = cv2.CALIB_CB_NORMALIZE_IMAGE+\
                                                cv2.CALIB_CB_ADAPTIVE_THRESH+\
                                                cv2.CALIB_CB_FILTER_QUADS)
    if ret1 and ret2:
        corners1 = cv2.cornerSubPix(img1,corners1,(5,5),(-1,-1),criteria)
        corners2 = cv2.cornerSubPix(img2,corners2,(5,5),(-1,-1),criteria)
        return(corners1,corners2)
    else:
        return (None,None)

def obtener_numero(img_path):
    nombre = os.path.basename(img_path)
    return int(nombre.split('.')[0].split('_')[-1])


print('Incio de la calibración\n')
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((pattern_size[0]*pattern_size[1],3), np.float32)
objp[:,:2] = np.mgrid[0:pattern_size[0],0:pattern_size[1]].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints1 = []
imgpoints2 = []
imgs_num= []


imgs_izq_list = sorted(glob(imgs_izq_dir+'/*'))
imgs_der_list = sorted(glob(imgs_der_dir+'/*'))

imgs_list_val = []
print(f'Obteniendo imágenes de {imgs_dir}')

for img_name_izq, img_name_der in zip(imgs_izq_list, imgs_der_list):
    print(f'\tObteniendo puntos de {os.path.basename(img_name_izq)} y {os.path.basename(img_name_izq)}')
    assert obtener_numero(img_name_izq) == obtener_numero(img_name_der), 'Error en el emparejamiento'
    img1 = cv2.imread(img_name_izq, 0)
    img2 = cv2.imread(img_name_der, 0)
    corners1, corners2 = buscar_y_extraer(img1, img2)
    if corners1 is not None:
        print('\t\tEncontrados... Guardando...')
        imgpoints1.append(corners1)
        imgpoints2.append(corners2)
        objpoints.append(objp)
        imgs_list_val.append([img_name_izq, img_name_der])
        imgs_num.append(obtener_numero(img_name_izq))

print(f'Se obtuvieron {len(imgs_list_val)} puntos de imágenes válidas')

imgpoints1_iter = imgpoints1[::]
imgpoints2_iter = imgpoints2[::]
objpoints_iter = objpoints[::]
imgs_index = list(range(len(objpoints)))

error_prom_max = np.inf
contador_calibs = 0
while error_prom_max>error_promedio_min:
    print(f'\nCalibrando cámaras, iteracion {contador_calibs+1}, {len(objpoints_iter)} imágenes')
    ## Calibración de cada una de las cámaras:
    print('\tCalibrando cámara izquierda...')
    ret1, M1, distcoef1, rvecs1, tvecs1, _, _, errors1 = cv2.calibrateCameraExtended(objpoints_iter, imgpoints1_iter,
                                                            img1.shape[::-1], None, None,
                                                            flags=cv2.CALIB_ZERO_TANGENT_DIST+\
                                                                  cv2.CALIB_FIX_K3)
    
    print(f'\t   -Error promedio {errors1.mean():.4}, max {errors1.max():.4}, min {errors1.min():.4}')
    print('\tCalibrando cámara derecha...')
    ret2, M2, distcoef2, rvecs2, tvecs2, _, _, errors2 = cv2.calibrateCameraExtended(objpoints_iter, imgpoints2_iter,
                                                            img2.shape[::-1], None, None,
                                                            flags=cv2.CALIB_ZERO_TANGENT_DIST+\
                                                                  cv2.CALIB_FIX_K3)
    print(f'\t   -Error promedio {errors2.mean():.4}, max {errors2.max():.4}, min {errors2.min():.4}')
    print('\tCalibrando sistema estéreo...')
    ret, M1, distcoef1, M2, distcoef2, R, T, E, F, perViewErrors =\
            cv2.stereoCalibrateExtended(objpoints_iter, imgpoints1_iter, imgpoints2_iter,
                                        M1, distcoef1, M2, distcoef2,
                                        img1.shape[::-1], None, None,
                                        criteria=criteria,
                                        flags = cv2.CALIB_FIX_INTRINSIC +\
                                                cv2.CALIB_SAME_FOCAL_LENGTH +\
                                                cv2.CALIB_ZERO_TANGENT_DIST+\
                                                cv2.CALIB_FIX_PRINCIPAL_POINT+\
                                                cv2.CALIB_FIX_ASPECT_RATIO+\
                                                cv2.CALIB_FIX_K3)
    #print(f'LOG perviewerrors {perViewErrors.shape}\n{perViewErrors}')
    promedio_errores = perViewErrors.mean(axis=0)
    print(f'\tError promedio\n\t\tCamara 1 = {promedio_errores[0]}'
         f'\n\t\tCamara 2 = {promedio_errores[1]}')
    
    error_prom_max = np.max(promedio_errores)
    umbral_errores = error_prom_max*factor_de_disminucion

    if  error_prom_max>error_promedio_min and len(objpoints_iter)-n_imgs_min>0:
        validos = np.logical_not(np.any(perViewErrors>umbral_errores,
                                 axis=1))
        indices_val = np.where(validos)[0]
        #Se sortean de menor a mayor para usar pop al eliminar
        indices_peores = np.argsort(np.mean(perViewErrors, axis=1))[::-1]
        indices_a_eliminar = [i for i in indices_peores if not validos[i]][:len(objpoints_iter)-n_imgs_min]
    else:
        break


    print(f'\t Se eliminarán {len(indices_a_eliminar)} imágenes. '
          f'Sus números son {", ".join([str(imgs_num[i]) for i in indices_a_eliminar])}')

    _indices = list(range(len(objpoints_iter)))
    imgpoints1_iter = [imgpoints1_iter[i] for i in _indices if i not in indices_a_eliminar]
    imgpoints2_iter = [imgpoints2_iter[i] for i in _indices if i not in indices_a_eliminar]
    objpoints_iter  = [objpoints_iter[i] for i in _indices if i not in indices_a_eliminar]
    imgs_num        = [imgs_num[i] for i in _indices if i not in indices_a_eliminar]
    imgs_index      = [imgs_index[i] for i in _indices if i not in indices_a_eliminar]
    print('\t Eliminadas correctamente.')
    contador_calibs += 1

print(f'\nError máximo alcanzado: {error_prom_max}. Número de patrones {len(objpoints_iter)}')

#Se grafican los errores
ancho = 0.4
x = np.arange(perViewErrors.shape[0])
fig, ax = plt.subplots()
ax.bar(x-ancho/2, perViewErrors[:,0], ancho, label='Cam1')
ax.bar(x+ancho/2, perViewErrors[:,1], ancho, label='Cam2')
ax.plot([0,x[-1]], [error_promedio_min, error_promedio_min], 'g', label='Error esperado')
ax.plot([0,x[-1]], [error_prom_max, error_prom_max], 'r', label='Error promedio')
ax.set_xticks(x)
ax.set_xticklabels(imgs_num)
ax.legend()
plt.show()


res = {'cameraMatrix1': M1,
       'cameraMatrix2': M2,
       'distCoeffs1': distcoef1,
       'distCoeffs2': distcoef2,
       'R': R,
       'T': T,
       'E': E,
       'F': F,
       'tamano_cuadro': args.tamano_cuadro}
print('\nGuardando resultados...')
save_filename = os.path.join(save_dir,'calibData.pk')
with open(save_filename, 'wb') as f:
    pickle.dump(res, f)
print('Los resultados fueron guardados en %s'%save_filename)
print('\nGuardando imágenes de reproyecciones...')
for indice in imgs_index:
    im1_name = os.path.basename(imgs_list_val[indice][0])
    im2_name = os.path.basename(imgs_list_val[indice][0])
    img1 = cv2.imread(imgs_izq_list[indice])
    img2 = cv2.imread(imgs_der_list[indice])
    cv2.drawChessboardCorners(img1, pattern_size, imgpoints1[indice], True)
    cv2.drawChessboardCorners(img2, pattern_size, imgpoints2[indice], True)

    if not os.path.exists(os.path.join(save_dir, 'calib_proyections')):
        os.makedirs(os.path.join(save_dir, 'calib_proyections', 'izq'))
        os.makedirs(os.path.join(save_dir, 'calib_proyections', 'der'))

    filename_im1 = os.path.join(save_dir, 'calib_proyections', 'izq', f'projected_{im1_name.split(".")[0]}.png')
    filename_im2 = os.path.join(save_dir, 'calib_proyections', 'der', f'projected_{im2_name.split(".")[0]}.png')
    cv2.imwrite(filename_im1, img1)
    cv2.imwrite(filename_im2, img2)
print(f'Imágenes guardadas en {os.path.join(save_dir, "calib_proyections")}')
