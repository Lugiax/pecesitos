from images_utils import *
import tensorflow as tf
import numpy as np
import os
import argparse

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str,
                    help='Directorio base donde se encuentran las imágenes y las máscaras')
parser.add_argument('name', type=str,
                    help='Nombre de la corrida')
parser.add_argument('--n_imgs', type=int,
                    help='Numero de imágenes de peces a utilizar para el entrenamiento',
                    default=1000)
parser.add_argument('--epocas', type=int,
                    help='Número de épocas para el entrenamiento',
                    default=10000)
parser.add_argument('--batch', type=int,
                    help='Tamaño del lote de entrenamiento ', default=8)
parser.add_argument('--plot', type=bool,
                    help='Mostrar o no los gráficos del entrenamiento', default=False)
parser.add_argument('--semillas', type=str,
                    help='Tipo de semillas a usar: "gauss" o ""', default='')
parser.add_argument('--save_dir', type=str,
                    help='Directorio base donde se descargarán todos los archivos',
                    default='.')
args = parser.parse_args()



print('Usando GPU',tf.config.experimental.list_physical_devices('GPU'))




## PARÁMETROS -----------------------------------------------------------------
CANALES = 14
DIM = 50
BATCH_SIZE = args.batch
N_IMGS = args.n_imgs #Número de imágenes para entrenar
EPOCAS = args.epocas
#Número de iteraciones en la ejecución del autómata
ITER_MIN = 50
ITER_MAX = 90
#Padding en las imágenes a utilizar
PADDING = 10
SEED=19

DATA_DIR = args.data_dir
CALIDAD_IMGS = 0.9

SAVE_DIR = args.save_dir
RUN_NAME = args.name

SHOW_PLOTS=args.plot
TIPO_SEMILLAS = args.semillas

## FUNCIONES DE APOYO --------------------------------------------------------

def obtener_vivos(x, umbral = 0.1):
    """
    Se considera que el primer canal es de la máscara y se busca en un
    vecindario inmediato las celdas que tengan al menos un vecino con 
    un valor mayor al umbral
    """
    mascara = x[..., :1]
    vivos = tf.nn.max_pool2d(mascara, 3, [1,1,1,1], 'SAME') > umbral
    return vivos

def generar_semillas(BATCH, DIM=DIM, CANALES=CANALES, tipo='gauss'):
    """
    Genera un arreglo de dimensiones (BATCH,DIM,DIM,CANALES) donde el primer canal
    es una máscara no binaria de distribución gaussiana en 2D con media en el
    centro del arreglo. El resto de canales tiene un 1 en el centro.
    """
    if tipo=='gauss':
        def gauss(x, mu=0.5, sigma=0.2):
            return np.exp(-(x-mu)**2/(2*sigma**2))

        x = np.repeat(gauss(np.linspace(0,1,DIM), sigma=0.1)[:, None],
                    DIM, 1).T
        y = np.repeat(gauss(np.linspace(0,1,DIM), sigma=0.1)[:, None],
                    DIM, 1)

        prod =np.matmul(y,x).astype(np.float32)
        semilla = np.concatenate((prod[..., None],
                                  np.zeros((DIM, DIM, CANALES-1),
                                dtype=np.float32)),
                                2)/prod.max()
        semilla[DIM//2, DIM//2, 1:] = 1
    else:
        semilla = np.zeros((DIM, DIM, CANALES), dtype=np.float32)
        semilla[DIM//2, DIM//2, :] = 1

    return tf.repeat(semilla[tf.newaxis, ...], BATCH, 0)

def f_perdida(x, mascara):
    """
    Función de pérdida para evaluar la calidad de las máscaras
    """
    resta = x[...,:1] - mascara[..., None]

    ##No termino de entender lo de los indices negativos :( 
    return tf.reduce_mean(tf.square(resta), [-2, -3, -1])


## MODELO Y FUNCIONES DE ENTRENAMIENTO ---------------------------------------


class Modelo(tf.keras.Model):
    def __init__(self, canales = CANALES):
        super().__init__()
        self.canales = canales
        self.nn = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(256, 1, activation = tf.nn.relu),
                    tf.keras.layers.Conv2D(self.canales, 1, activation=None,
                                           #kernel_initializer=tf.zeros_initializer
                                           )
                  ])
        ##Una llamada dummy para construir el modelo
        self(tf.zeros((1, 3, 3, self.canales)), tf.zeros((3,3,3)))

    @tf.function
    def percibir(self, x, imagen):
        """
        Función de percepción. Evalúa los gradientes en el eje x, el eje y y concatena
        los gradientes obtenidos con el estado actual del autómata. Además, antes de
        la evaluación, se agregan los canales de la imagen al estado actual, para que
        actúen como ambiente.
        """
        BATCH_SIZE = x.shape[0]

        identidad = np.outer(np.float32([0,1,0]), np.float32([0,1,0]))
        dx = np.float32(np.outer([1,2,1], [-1,0,1])/8.)
        dy = np.transpose(dx)
        kernel = tf.stack([identidad, dx, dy], -1)[:, :, tf.newaxis, :]
        kernel = tf.repeat(kernel, self.canales+3, 2)

        img_batch = tf.repeat(imagen[None, ...], BATCH_SIZE, 0)
        x_concat = tf.concat((x, img_batch), 3)
        
        return tf.nn.depthwise_conv2d(x_concat, kernel, [1,1,1,1], 'SAME')
    
    @tf.function
    def call(self, x, imagen, paso=0.1, umbral_actualizacion=0.5):
        """
        Se hace un paso en el autómata. Se revisan las celulas vivas, se efectúa la percepcíon,
        luego la regla de actualización. Finalmente se revisa las células que "nacieron" o que 
        "murieron" y se actualizan sus estados.
        """
        vivos_antes = obtener_vivos(x)

        percepcion = self.percibir(x, imagen)
        dx = self.nn(percepcion)*paso

        mascara_actualizacion = tf.random.uniform(tf.shape(x[:, :, :, :1])) <= umbral_actualizacion
        x += dx*tf.cast(mascara_actualizacion, tf.float32)

        vivos_despues = obtener_vivos(x)
        vivos = tf.logical_and(vivos_antes, vivos_despues)

        return  x*tf.cast(vivos, tf.float32)


#            ***********************************************************************************
## EJECUCIÓN -----------------------------------------------------------------------------------
#            ***********************************************************************************


#Se obtienen los datos de las máscaras disponibles y se generan las regiones con las que se
#trabajarán

print('Generando las regiones')
masks_data = mascaras_disponibles(DATA_DIR, umbral_calidad=CALIDAD_IMGS)
regiones = generar_regiones(DATA_DIR, masks_data, N_IMGS,
                            solo_horizontales=True,
                            resize_shape=(DIM,DIM), padding=PADDING,
                            seed=SEED)
if len(regiones)!= N_IMGS:
    print(f'No hay suficientes imágenes como las solicitadas. {len(regiones)} disponibles')
else:
    print('Regiones generadas satisfactoriamente :D')

#Se hace un diccionario con las imágenes y se convierten a tensores
datos = {i: [tf.constant(img, dtype=tf.float32),
             tf.constant(mask, dtype=tf.float32)] for i, (img, mask) in enumerate(regiones)}



#definición del modelo y de la función de entrenamiento
modelo = Modelo()
print(modelo.summary())

lr = 2e-3
lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    [2000], [lr, lr*0.1])
trainer = tf.keras.optimizers.Adam(lr_sched)

@tf.function
def paso_entrenamiento(x, imagen, mascara):
    niter = tf.random.uniform([], ITER_MIN, ITER_MAX, tf.int32)
    with tf.GradientTape() as g:
        for i in tf.range(niter):
            x = modelo(x, imagen)
        perdida = tf.reduce_mean(f_perdida(x, mascara))
    gradientes = [grad/(tf.norm(grad)+1e-8) for grad in\
                  g.gradient(perdida, modelo.weights)]
    trainer.apply_gradients(zip(gradientes, modelo.weights))

    return x, perdida


#Se crean los directorios para guardar las corridas
if not os.path.isdir(os.path.join(SAVE_DIR, RUN_NAME)):
    folder = os.path.join(SAVE_DIR, RUN_NAME)
    os.makedirs(os.path.join(folder, 'weights'))
    os.makedirs(os.path.join(folder, 'res_imgs'))
    for i in range(N_IMGS):
        os.mkdir(os.path.join(folder, 'res_imgs', f'{i}'))
pesos_filename = os.path.join(SAVE_DIR, RUN_NAME, 'weights',f'pesos_{RUN_NAME}.tf')


perdidas_log = {i:[] for i in range(N_IMGS)}

for e in range (EPOCAS+1):
    id_img = random.randint(0, N_IMGS-1)
    print(f'Paso {e+1}, imagen {id_img}')

    imagen, mascara = datos[id_img]
    ##Tal vez sería bueno hacer que las semillas se parecieran más a lo 
    ##del artículo. Aquí se generan nuevas semillas en cada época
    x0 = generar_semillas(BATCH_SIZE, DIM=DIM, tipo=TIPO_SEMILLAS)

    x , perdida = paso_entrenamiento(x0, imagen, mascara)
    perdidas_log[id_img].append(perdida)

    if e%5==0:
        ids = f_perdida(x, mascara).numpy().argsort()
        mejor = x[ids[0]]
        peor = x[ids[-1]]

        fig, axes = plt.subplots(1, 4, figsize=(12,4))
        imgs_to_plot = [imagen, mascara, mejor[..., 0], peor[..., 0]]
        titulos = ['Original', 'Original', 'Mejor', 'Peor']
        for ax, img, titulo in zip(axes, imgs_to_plot, titulos):
            ax.imshow(img)
            ax.set_title('Original')
            ax.axis('off')
        if SHOW_PLOTS: plt.show()

        im_name = f'epoca-{e:0>6}.png'
        plt.savefig(os.path.join(SAVE_DIR, RUN_NAME, 'res_imgs',
                                 str(id_img), im_name))

        fig, ax = plt.subplots()
        ax.plot(perdidas_log[id_img])
        ax.set_title('Evolución de la\nfunción de pérdida')
        ax.set_xlabel('Época')
        ax.set_ylabel('Error')
        if SHOW_PLOTS: plt.show()

        plt.savefig(os.path.join(SAVE_DIR, RUN_NAME, 'res_imgs',
                                 str(id_img),'perdida.png'))
    
    if e%100 == 0:
        modelo.save_weights(os.path.join(SAVE_DIR, RUN_NAME, 'weights',f'pesos_{RUN_NAME}.tf'), save_format='tf')

    print(f'\t Perdida: {perdida}')

modelo.save_weights(pesos_filename, save_format='tf')
print(f'Fin del entrenamiento. Pesos guardados en ')
