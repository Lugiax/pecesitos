import os, csv, random
from glob import glob
from skimage.io import imread
from skimage.transform import resize
from skimage.measure import regionprops_table
from skimage.color import gray2rgb
import numpy as np

def obtener_codigos_clases(archivo):
    """	
    <- archivo: class-descriptions-boxable.csv obtenido al decargar la base de datos
    -> codigos: diccionario con cada nombre y su respectivo código
    """
    with open(archivo, 'r') as f:
        codigos={}
        lineas = csv.reader(f)
        for l in lineas:
            codigos[l[1].lower()] = l[0]
    return codigos

def mascaras_disponibles(data_dir, clases=['fish'], f_clases='class-descriptions-boxable.csv',
                        f_mascaras='train-annotations-object-segmentation.csv',
                        masks_dir_name='masks', umbral_calidad=0.9):
    """
    data_dir: directorio donde se descargaron los datos de las imágenes y las máscaras
    clases: lista con los nombres de las clases de interés
    masks_dir_name: nombre de la carpeta de máscaras del data_dir
    umbral_calidad: umbral según la calificación de la calidad de las máscaras

    mask_data: diccionario con las rutas a las máscaras disponibles para cada clase
    """
    codigos = obtener_codigos_clases(os.path.join(data_dir, f_clases))
    validos = {codigos[c.lower()]:c.lower() for c in clases}
    masks_dir = os.path.join(data_dir, masks_dir_name)

    mask_data = {}#{c:{} for c in clases}
    with open(os.path.join(data_dir, f_mascaras), 'r') as f:
        lineas = csv.reader(f)
        lineas.__next__()#Se quita el encabezado
        for l in lineas:
            if l[2] in list(validos.keys()) and\
            os.path.exists(os.path.join(masks_dir, l[0])) and\
            float(l[8])>=umbral_calidad:
                #clase = validos[l[2]]
                try:
                    #mask_data[clase][l[1]].append(l[0])
                    mask_data[l[1]].append(l[0])
                except:
                    #mask_data[clase][l[1]] = [l[0]]
                    mask_data[l[1]] = [l[0]]

    return mask_data

def get_bboxes(f_path, shape):
    """
    fpath:  ruta al archivo de las localizaciones
            el formato debe ser:
                clase, centro_x, centro_y, ancho, alto
    shape: Forma de la imagen

    bboxes: Lista con las coordenadas de las bounding boxes
            encontradas en el archivo. Cada bounding box es 
            una lista con el formato:
                [y1, y2, x1, x2]
    """
    bboxes = []
    H, W = shape[:2]
    with open(f_path, 'r') as f:
        lineas = csv.reader(f, delimiter=' ')
        for l in lineas:
            _, bxc, byc, bw, bh = [float(i) for i in l]
            xc, w2 = int(bxc*W), int(bw*W/2)
            yc, h2 = int(byc*H), int(bh*H/2) 
            bboxes.append([yc-h2, yc+h2+1,
                           xc-w2, xc+w2+1])
    return bboxes


def generar_regiones(data_dir, masks_disponibles, N, aleatorio=True,
                    res_min=(100,100), solo_horizontales=True,
                    resize_shape=None,  padding=0, seed=None):
    """
    Genera imágenes que corresponden a regiones de segmentaciones en
    imágenes. En una imagen puede existir más de un objeto deseado y
    por lo tanto una región por cada uno de ellos. Esta función hace 
    la búsqueda en las imágenes con  máscaras disponibles y devuelve
    N regiones (imagen, mascara) válidas de resolución mínima especi-
    ficada.
    """

    root = os.path.abspath(data_dir)
    imgs_dir = os.path.join(root, 'fish','images')
    masks_dir = os.path.join(root, 'masks')

    lista_imgs_paths = glob(os.path.join(imgs_dir, '*'))
    if seed is not None: random.seed(seed)
    if aleatorio: random.shuffle(lista_imgs_paths)

    restantes = N
    regiones = []
    while restantes*len(lista_imgs_paths) > 0:
        siguiente = lista_imgs_paths.pop()
        img_id = os.path.basename(siguiente).split('.')[0]
        if img_id not in masks_disponibles.keys():
            continue
        img = imread(siguiente)
        if len(img.shape)<3: #en caso de tener una imagen en grises
            img = gray2rgb(img)
        H,W = img.shape[:2]

        masks_paths = masks_disponibles[img_id]
        for mp in masks_paths:
            mask = resize(imread(os.path.join(masks_dir,mp)),
                          (H,W))
            bmask = (mask>0).astype(int)

            reg_props = regionprops_table(bmask, properties=('bbox',))
            y1, x1, y2, x2 = [reg_props[f'bbox-{i}'][0] for i in range(4)]
            ##Se aplica el padding y se hace un ajuste
            y1 = max(0, y1-padding)
            y2 = min(H-1, y2+padding)
            x1 = max(0, x1-padding)
            x2 = min(W-1, x2+padding)
            res_y = y2-y1
            res_x = x2-x1

            #Se revisa la resolución mínima de las regiones
            if res_y<res_min[0] or res_x<res_min[1]:
                continue
            #Se revisa que sean horizontales
            if solo_horizontales and res_y*1.3>res_x:
                continue
            reg_img = np.float32(img[y1:y2, x1:x2]/255)
            reg_mask = np.float32(bmask[y1:y2, x1:x2])

            if resize_shape is not None:
                assert len(resize_shape)==2
                reg_img = resize(reg_img, resize_shape)
                reg_mask =resize(reg_mask,resize_shape)

            regiones.append([reg_img,  #region RGB
                            reg_mask]) #region Mask
            restantes-=1
            if restantes == 0:
                break
    
    return regiones 