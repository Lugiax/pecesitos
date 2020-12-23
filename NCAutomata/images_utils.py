import os, csv, random


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
                        masks_dir_name='masks', umbral_calidad=0.8):
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