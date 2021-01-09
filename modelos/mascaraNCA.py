import tensorflow as tf
import numpy as np
from images_utils import*
import matplotlib.pyplot as plt
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

###Modelo ---------------------------------------------------------------------

class NCA(tf.keras.Model):
    def __init__(self, canales=14, dim=100):
        super().__init__()
        self.iter_min = 50
        self.iter_max = 90
        self.dim = dim
        self.lr=2e-3
        self.canales = canales
        self.nn = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(256, 1, activation = tf.nn.relu),
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
                seed=None, save_dir='./modelo', run_name='run1', plot=False):
        print('Generando las regiones')
        if dim is None:
            dim = self.dim
        else:
            self.dim = dim
        masks_data = mascaras_disponibles(DATA_DIR, umbral_calidad=calidad_imgs)
        regiones = generar_regiones(DATA_DIR, masks_data, n_imgs,
                                    solo_horizontales=True,
                                    resize_shape=(dim,dim), padding=padding,
                                    seed=seed)
        assert len(regiones)==n_imgs, f'Solo hay {len(regiones)} imágenes disponibles'

        datos = {i:[tf.constant(img, dtype=tf.float32),
                   tf.constant(mask, dtype=tf.float32)] for i, (img, mask) in enumerate(regiones)}

        RUN_DIR = os.path.join(save_dir, run_name)
        if not os.path.isdir(RUN_DIR):
            os.makedirs(os.path.join(RUN_DIR, 'weights'))
            os.makedirs(os.path.join(RUN_DIR, 'res_imgs'))
            for i in range(n_imgs):
                os.mkdir(os.path.join(RUN_DIR, 'res_imgs', f'{i}'))

        perdidas_log = {i:[] for i in range(n_imgs)}

        for e in range (epocas+1):
            id_img = random.randint(0, n_imgs-1)
            print(f'Paso {e+1}, imagen {id_img}')

            imagen, mascara = datos[id_img]
            ##Tal vez sería bueno hacer que las semillas se parecieran más a lo 
            ##del artículo. Aquí se generan nuevas semillas en cada época
            x0 = generar_semillas(batch, CANALES=self.canales, DIM=dim, tipo='')

            x , perdida = self.paso_entrenamiento(x0, imagen, mascara)
            perdidas_log[id_img].append(perdida)

            if e%int(0.05*epocas)==0:
                ids = f_perdida(x, mascara).numpy().argsort()
                mejor = x[ids[0]]
                peor = x[ids[-1]]

                fig, axes = plt.subplots(1, 4, figsize=(20,8))
                imgs_to_plot = [imagen, mascara, mejor[..., 0], peor[..., 0]]
                titulos = ['Original', 'Original', 'Mejor', 'Peor']
                for ax, img, titulo in zip(axes, imgs_to_plot, titulos):
                    ax.imshow(img)
                    ax.set_title(titulo)
                    ax.axis('off')
                if plot: plt.show()

                im_name = f'epoca-{e:0>6}.png'
                plt.savefig(os.path.join(RUN_DIR, 'res_imgs',
                                         str(id_img), im_name))

                fig, ax = plt.subplots()
                ax.plot(perdidas_log[id_img])
                ax.set_title('Evolución de la\nfunción de pérdida')
                ax.set_xlabel('Época')
                ax.set_ylabel('Error')
                if plot: plt.show()

                plt.savefig(os.path.join(RUN_DIR, 'res_imgs',
                                         str(id_img),'perdida.png'))
            
            if e%int(0.2*epocas) == 0:
                self.save_weights(os.path.join(RUN_DIR, 'weights'), save_format='tf')

            print(f'\t Perdida: {perdida}')

        self.save_weights(os.path.join(RUN_DIR, 'weights'), save_format='tf')
        print(f'Fin del entrenamiento. Pesos guardados en {os.path.join(RUN_DIR,"weights")}')

    def cargar_pesos(self, pesos_path):
        self.load_weights(pesos_path)

    def generar(self, img0, batch=1, iteraciones=None, canales=None, dim=None,
                weights_path=None, bin_mask=False, umbral=0.5):
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

        img = img0[None, ...]/255
        img = tf.image.resize(img.astype(np.float32), (dim,dim), antialias=True)
        mask =  generar_semillas(batch, dim, canales)
        print('Iniciando inferencia')
        for _ in tf.range(iteraciones):
            mask = self(mask, img[0])

        res_mask = tf.image.resize(mask[..., :1], img0.shape[:2])
        if bin_mask:
            return obtener_vivos(res_mask, umbral)[..., 0]
        else:
            return res_mask[..., 0]




if __name__=='__main__':
    from skimage.io import imread
    DATA_DIR = '/home/carlos/Documentos/Codes/ProyectoMCC/datos'
    SAVE_DIR = '/home/carlos/Documentos/Codes/ProyectoMCC/corridas'
    RUN_NAME = 'prueba_BORRAR'

    img_path = ''
    w_path = ''
    automata = NCA()
    """automata.entrenar(DATA_DIR,
                      epocas=100,
                      dim=50,
                      batch=4,
                      n_imgs=1,
                      padding=10,
                      save_dir=SAVE_DIR,
                      run_name=RUN_NAME)
    """
    automata.load_weights(w_path)

    img = imread(img_path)
    res = automata.generar(img)
    print(res.shape)
