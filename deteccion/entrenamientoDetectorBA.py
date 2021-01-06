import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imread
from skimage.color import gray2rgb
from random import sample
import random
from glob import glob
import argparse

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

parser = argparse.ArgumentParser()
parser.add_argument('imgs_dir', type=str,
                    help='Directorio donde se encuentran las imágenes')
parser.add_argument('anot_dir', type=str,
                    help='Directorio donde se encuentran las anotaciones')
parser.add_argument('--dim', type=int,
                    help='Dimension de las imagenes procesadas',
                    default=100)
parser.add_argument('--epocas', type=int,
                    help='Número de épocas para el entrenamiento',
                    default=10000)
parser.add_argument('--batch', type=int,
                    help='Tamaño del lote de entrenamiento ', default=16)
parser.add_argument('--save_dir', type=str,
                    help='Directorio base donde se guardará el modelo',
                    default='.')
parser.add_argument('--nombre', type=str,
                    help='Nombre del modelo')
args = parser.parse_args()



#-----------------------------------------------------------------------

def obtener_id(path):
    return os.path.basename(path).split('.')[0]

def obtener_datos(anot_path, imgs_dir, resized_shape = (50,50)):
    img_id = obtener_id(anot_path)
    img = imread(os.path.join(imgs_dir, img_id+'.jpg'))
    regiones = []
    with open(anot_path, 'r') as f:
        reader = csv.reader(f)
        for r in reader:
            x1,x2,y1,y2 = [int(v) for v in r[:4]]
            H,W= resized_shape
            p1y,p1x,p2y,p2x = [float(v) for v in r[4:]]
            regiones.append( ( resize(img[y1:y2, x1:x2], resized_shape),
                              [p1y,p1x,p2y,p2x] ) )
    return regiones

def generar_batch(lista, imgs_dir, N, DIM):
    por_revisar = sample(lista, N)  
    imgs = []
    pts = []
    while N>0:
        for img, pt in obtener_datos(por_revisar.pop(), imgs_dir, resized_shape=(DIM,DIM)):
            if len(img.shape)==2:
                img = gray2rgb(img)
            imgs.append(img)
            pts.append(pt)
            N-=1
    return np.array(imgs, dtype=np.float32), np.array(pts, dtype=np.float32)

#-----------------------------------------------------------------------

class Modelo(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.nn = tf.keras.Sequential([
                    Conv2D(16, (3,3), activation = tf.nn.relu),
                    MaxPooling2D((2,2)),
                    Conv2D(32, (3,3), activation = tf.nn.relu),
                    MaxPooling2D((2,2)),
                    Conv2D(64, (3,3), activation = tf.nn.relu),
                    MaxPooling2D((2,2)),
                    Conv2D(128, (3,3), activation = tf.nn.relu),
                    Flatten(),
                    Dense(64, activation = tf.nn.relu),
                    Dense(32, activation = tf.nn.relu),
                    Dense(4)
                  ])
    @tf.function
    def call(self, x):
        return self.nn(x)


def f_error(y_t,y_p):
    d1 = tf.norm(y_p[:, :2] - y_t[:, :2], ord='euclidean', axis=1)
    d2 = tf.norm(y_p[:, 2:] - y_t[:, 2:], ord='euclidean', axis=1)
    ds = tf.concat((d1[tf.newaxis, :],d2[tf.newaxis, :]), axis=0)
    return tf.reduce_mean(ds)

#----------------------------------------------------------------------

BATCH = args.batch
DIM = args.dim
EPOCAS = args.epocas
SAVE_DIR = args.save_dir
RUN_NAME = args.nombre
ANOT_DIR = args.anot_dir
IMGS_DIR = args.imgs_dir
P_TRAIN = 0.85


RUN_DIR = os.path.join(SAVE_DIR, RUN_NAME)
if not os.path.isdir(RUN_DIR):
	os.makedirs(RUN_DIR)

ANOT_LIST = glob(os.path.join(ANOT_DIR, '*.txt'))
random.shuffle(ANOT_LIST)
train_list = ANOT_LIST[:int(P_TRAIN*len(ANOT_LIST))]
test_list = ANOT_LIST[int(P_TRAIN*len(ANOT_LIST)):]

modelo = Modelo()

lr = 2e-3
lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    [2000], [lr, lr*0.1])
trainer = tf.keras.optimizers.Adam(lr_sched)

@tf.function
def paso_entrenamiento(x, y):
    with tf.GradientTape() as g:
        yp = modelo(x)
        perdida = f_error(y, yp)
    gradientes = [grad/(tf.norm(grad)+1e-8) for grad in\
                  g.gradient(perdida, modelo.weights)]
    trainer.apply_gradients(zip(gradientes, modelo.weights))

    return perdida

LOG = []

for i in range(EPOCAS):
    #Obtener los datos de entrenamiento
    x, y = generar_batch(train_list, IMGS_DIR, BATCH, DIM)
    perdida = paso_entrenamiento(x, y)
    LOG.append(perdida)
    print(f'Paso {i}. Pérdida {perdida}')
    if i%10==0:
        plt.plot(LOG)
        plt.show()
    if i%100 == 0:
        modelo.save(RUN_DIR)
        plt.savefig(os.path.join(RUN_DIR, 'perdida.png'))

#Evaluación
metrica = tf.keras.metrics.RootMeanSquaredError()
imgs_lote, pts_lote = generar_batch(test_list, IMGS_DIR, len(test_list), DIM)
prediccion = modelo(imgs_lote)
metrica.update_state(pts_lote, prediccion)


print(f'El resultado RMSE (Root Mean Squared Error) = {metrica.result().numpy()}')

modelo.save(RUN_DIR)
plt.plot(LOG)
plt.savefig(os.path.join(RUN_DIR, 'perdida.png'))
