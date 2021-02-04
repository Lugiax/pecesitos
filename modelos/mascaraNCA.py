import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, sys, cv2
from skimage.transform import resize
if __name__=='__main__':
    from ..herramientas.general import Grabador
else:
    from herramientas.general import Grabador

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
from images_utils import*
#tf.config.experimental.list_physical_devices('GPU')

###Funciones-----------------------------------------------------------------------
def obtener_vivos(x, umbral = 0.1):
    #Se considera que el primer canal es de la máscara y se busca en un
    #vecindario inmediato las celdas que tengan al menos un vecino con 
    #un valor mayor al umbral
    mascara = x[..., :1]
    vivos = tf.nn.max_pool2d(mascara, 3, [1,1,1,1], 'SAME') > umbral
    return vivos

def generar_semillas(N, DIM=100, CANALES=14, tipo='gauss'):
    if tipo=='gauss':
        def gauss(x, mu=0.5, sigma=0.2):
            return np.exp(-(x-mu)**2/(2*sigma**2))

        x = np.repeat(gauss(np.linspace(0,1,DIM), sigma=0.1)[:, None],
                    DIM, 1).T
        y = np.repeat(gauss(np.linspace(0,1,DIM), sigma=0.1)[:, None],
                    DIM, 1)

        prod =np.matmul(y,x).astype(np.float32)
        semilla = np.concatenate((prod[..., None],
                                  np.zeros((DIM, DIM, CANALES-1), dtype=np.float32)
                                  ), 2)/prod.max()
        semilla[DIM//2, DIM//2, 1:] = 1
    else:
        semilla = np.zeros((DIM, DIM, CANALES), dtype=np.float32)
        semilla[DIM//2, DIM//2, :] = 1

    return tf.repeat(semilla[tf.newaxis, ...], N, 0)

def f_perdida(x, mascara):
    ##No termino de entender lo de los indices negativos :( 
    resta = x[...,:1] - mascara[..., None]
    return tf.reduce_mean(tf.square(resta), [-2, -3, -1])

def guardar_log(log, fname):
    with open(fname, 'w') as f:
        for k in log:
            f.write(f'{k},{",".join([str(i) for i in log[k]])}\n')

###Modelo ---------------------------------------------------------------------

class NCA(tf.keras.Model):
    def __init__(self, canales=14, dim=100):
        super().__init__()
        self.iter_min = 1
        self.iter_max = 5
        self.dim = dim
        self.lr=2e-3
        self.canales = canales
        self.nn = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(128, 1, activation = tf.nn.relu),
                    tf.keras.layers.Conv2D(self.canales, 1, activation=None,
                                           #kernel_initializer=tf.zeros_initializer
                                           )
                  ])
        self.trainer = tf.keras.optimizers.Adam(
                        tf.keras.optimizers.schedules.PiecewiseConstantDecay([2000],
                                                                             [self.lr, self.lr*0.1])
                        )
        ##Una llamada dummy para construir el modelo
        self(tf.zeros((1, 3, 3, self.canales)), tf.zeros((3,3,3)))
    @tf.function
    def percibir(self, x, imagen):
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
        vivos_antes = obtener_vivos(x)

        percepcion = self.percibir(x, imagen)
        dx = self.nn(percepcion)*paso

        mascara_actualizacion = tf.random.uniform(tf.shape(x[:, :, :, :1])) <= umbral_actualizacion
        x += dx*tf.cast(mascara_actualizacion, tf.float32)

        vivos_despues = obtener_vivos(x)
        vivos = tf.logical_and(vivos_antes, vivos_despues)

        return  x*tf.cast(vivos, tf.float32)

    @tf.function
    def paso_entrenamiento(self, x, imagen, mascara):
        niter = tf.random.uniform([], self.iter_min, self.iter_max, tf.int32)
        with tf.GradientTape() as g:
            for i in tf.range(niter):
                x = self(x, imagen)
            perdida = tf.reduce_mean(f_perdida(x, mascara))
        gradientes = [grad/(tf.norm(grad)+1e-8) for grad in\
                      g.gradient(perdida, self.weights)]
        self.trainer.apply_gradients(zip(gradientes, self.weights))

        return x, perdida

    def entrenar(self, DATA_DIR, epocas=10000, batch=8, n_imgs=100, calidad_imgs=0.9, dim=None, padding=10,
                seed=None, save_dir='./modelo', run_name='run1', plot=False, prop_ent_test=0.8):
        print('Generando las regiones')
        if dim is None:
            dim = self.dim
        else:
            self.dim = dim
        #masks_data = mascaras_disponibles(DATA_DIR, umbral_calidad=calidad_imgs)
        regiones = generar_regiones(DATA_DIR, n_imgs,
                                    solo_horizontales=True,
                                    resize_shape=(dim,dim), padding=padding,
                                    seed=seed)
        assert len(regiones)==n_imgs, f'Solo hay {len(regiones)} imágenes disponibles'

        datos = {i:[tf.constant(img, dtype=tf.float32),
                   tf.constant(mask, dtype=tf.float32),
                   generar_semillas(batch, CANALES=self.canales, DIM=dim, tipo='')
                   ] for i, (img, mask) in enumerate(regiones)}

        #datos_ent = {i:datos[i] for i in list(datos.keys())[:int(n_imgs*prop_ent_test)]}

        RUN_DIR = os.path.join(save_dir, run_name)
        if not os.path.isdir(RUN_DIR):
            os.makedirs(os.path.join(RUN_DIR, 'res_imgs'))
            for i in range(n_imgs):
                os.mkdir(os.path.join(RUN_DIR, 'res_imgs', f'{i}'))

        with open(os.path.join(RUN_DIR, 'info.txt'), 'w') as f:
            f.write(f'epocas:{epocas}\nbatch:{batch}\nn_imgs:{n_imgs}\ndim:{dim}\npadding:{padding}')

        perdidas_log = {i:[] for i in range(n_imgs)}

        for e in range (epocas+1):
            id_img = random.randint(0, int(n_imgs*prop_ent_test))
            print(f'Paso {e+1}, imagen {id_img}')

            imagen, mascara, x0 = datos[id_img]
            ##Tal vez sería bueno hacer que las semillas se parecieran más a lo 
            ##del artículo. Aquí se generan nuevas semillas en cada época
            #x0 = generar_semillas(batch, CANALES=self.canales, DIM=dim, tipo='')


            x , perdida = self.paso_entrenamiento(x0, imagen, mascara)
            perdidas_log[id_img].append(float(perdida))

            #if e%1==0:
            ids = f_perdida(x, mascara).numpy().argsort()
            mejor = x[ids[0]]
            peor = x[ids[-1]]

            fig, axes = plt.subplots(2, 4, figsize=(20,8))
            imgs_to_plot = [x[i, ..., 0] for i in ids]
            #imgs_to_plot = [imagen, mascara, mejor[..., 0], peor[..., 0]]
            #titulos = ['Original', 'Original', 'Mejor', 'Peor']
            for ax, img in zip(axes.ravel(), imgs_to_plot):#, titulos):
                ax.imshow(img)
                #ax.set_title(titulo)
                ax.axis('off')
            #if plot: plt.show()

            im_name = f'epoca-{e:0>6}.png'
            plt.savefig(os.path.join(RUN_DIR, 'res_imgs',
                                        str(id_img), im_name))

            #Se actualiza el conjunto de estados
            x0 = x.numpy()
            x0[ids[-1]] = generar_semillas(1, CANALES=self.canales, DIM=dim, tipo='')[0]
            datos[id_img][2] = x0

            fig, ax = plt.subplots()
            ax.plot(perdidas_log[id_img])
            ax.set_title('Evolución de la\nfunción de pérdida')
            ax.set_xlabel('Época')
            ax.set_ylabel('Error')
            #if plot: plt.show()

            plt.savefig(os.path.join(RUN_DIR, 'res_imgs',
                                        str(id_img),'perdida.png'))
        
            #if e%int(0.2*epocas) == 0:
            #    self.save_weights(os.path.join(RUN_DIR, 'weights'), save_format='tf')
            #    guardar_log(perdidas_log, os.path.join(RUN_DIR, 'log.csv'))

            print(f'\t Perdida: {perdida}')

        self.save_weights(os.path.join(RUN_DIR, 'weights'), save_format='tf')
        guardar_log(perdidas_log, os.path.join(RUN_DIR, 'log.csv'))
        print(f'Fin del entrenamiento. Pesos guardados en {os.path.join(RUN_DIR,"weights")}')

        ##Imagenes de prueba
        log_errores=[]
        for i in list(datos.keys())[int(n_imgs*prop_ent_test):]:
            img, mask, _ = datos[i]
            mask_gen = tf.constant(self.generar(img.numpy(), not_bin=True)[None, ..., None], dtype=tf.float32)
            error = tf.reduce_mean(f_perdida(mask_gen, mask)).numpy()
            log_errores.append(error)
        print(f'El error promedio en el conjunto de prueba fue de {sum(log_errores)/len(log_errores)}')




    def cargar_pesos(self, pesos_path):
        self.load_weights(pesos_path)

    def generar(self, img0, batch=1, iteraciones=None, canales=None, dim=None,
                weights_path=None, not_bin=False, umbral=0.1, original_size=True,
                nombre_video=''):
        """
        imgs puede ser un arreglo o una lista de imágenes
        """
        if iteraciones is None:
            iteraciones = tf.random.uniform([], self.iter_min, self.iter_max, tf.int32)
        if canales is None:
            canales = self.canales
        if dim is None:
            dim = self.dim
        if weights_path is not None:
            self.load_weights(weights_path)
        normalizar=False
        if img0.max()>1:
            img = img0/255
            normalizar=True
        else:
            img = img0.copy()
        if nombre_video!='':
            grabador = Grabador(nombre_video, fps=5)

        img = resize(img, (dim,dim), anti_aliasing=True).astype(np.float32)
        mask =  generar_semillas(batch, dim, canales)

        for _ in tf.range(iteraciones):
            mask = self(mask, img)
            if nombre_video!='':
                frame = (np.clip(mask[0,...,0].numpy(), 0,1)*255).astype(np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                grabador.agregar(frame)

        if original_size:
            mask = tf.image.resize(mask[..., :1],
                                    img0.shape[:2],
                                    method='bicubic')
        if nombre_video!='':
            grabador.terminar() 
        
        if not_bin:
            return np.clip(mask[0, ..., 0].numpy(), 0, 1)
        else:
            return obtener_vivos(mask, umbral)[0,...,0].numpy() 




if __name__=='__main__':
    import argparse
    from PIL import Image

    from glob import glob
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('accion', type=str,
                        help='entrenar, inferir')
    parser.add_argument('dir', type=str,
                        help='Directorio de datos')
    parser.add_argument('--canales', type=int,
                        help='Número de canales de las semillas', 
                        default = 14)
    parser.add_argument('--epocas', type=int,
                        help='Número de épocas de entrenamiento', 
                        default = 10000)
    parser.add_argument('--dim', type=int,
                        help='Tamaño de las imágenes de entrenamiento',
                        default=100)
    parser.add_argument('--batch', type=int,
                        help='Número de máscaras por paso',
                        default=8)
    parser.add_argument('--n_imgs', type=int,
                        help='Número de imágenes para hacer el entrenamiento',
                        default=10)
    parser.add_argument('--padding', type=int,
                        help='Padding de las imágenes de entrenamiento',
                        default=10)
    parser.add_argument('--save_dir', type=str,
                        help='Directorio para guardar los resultados',
                        default='.')
    parser.add_argument('--run_name', type=str,
                        help='Nombre de la corrida',
                        default='corr1')
    parser.add_argument('--w_path', type=str,
                        help='Ruta a los pesos del modelo',
                        default='corr1/weights')
    args = parser.parse_args()


    automata = NCA()

    assert args.accion in ['entrenar', 'inferir'], 'Seleccionar una acción válida'
    if args.accion=='entrenar':
        automata.entrenar(args.dir,
                          epocas=args.epocas,
                          dim=args.dim,
                          batch=args.batch,
                          n_imgs=args.n_imgs,
                          padding=args.padding,
                          save_dir=args.save_dir,
                          run_name=args.run_name)
    else:
        automata.load_weights(args.w_path)
        imgs_list = glob(os.path.join(dir, '*.*'))
        for img_path in imgs_list:
            im_name = os.path.basename(img_path).split('.')[0]
            img = Image.open(img)
            res = Image.fromarray(automata.generar(img)[0])
            res.save(os.path.join(args.save_dir, im_name+'.png'))
