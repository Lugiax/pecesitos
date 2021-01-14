#from mascaraNCA import NCA
import cv2
from glob import glob
import os
import matplotlib.pyplot as plt
import csv
import numpy as np

def mostrar_animacion(imgs_dir, loop=True):
    imgs_dir = os.path.abspath(imgs_dir)
    numero = imgs_dir.split('/')[-1]
    formats = ('jpg', 'png')
    imgs_paths = [f for f in glob(os.path.join(imgs_dir,'*.*'))\
                  if f.split('.')[-1] in formats and 'epoca' in f]
    imgs_paths.sort()
    count = 0
    while True:
        siguiente = cv2.imread(imgs_paths[count])
        shape = siguiente.shape
        siguiente = cv2.resize(siguiente, (int(shape[1]*0.5), int(shape[0]*0.5)))
        siguiente = cv2.putText(siguiente, f'Secuencia {count}', (20,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0))
        cv2.imshow(f'Animacion de imagen {numero}', siguiente)
        k = cv2.waitKey(200)
        if k == ord('q'):
            break

        count+=1

        if count==len(imgs_paths) and loop:
            count = 0
        elif count==len(imgs_paths):
            break
    cv2.destroyWindow('Animacion')

def graficar_perdidas(logs_path, unicos=None, figsize=(10,5), n_mejores=5, n_peores=0, title=''):
    with open(logs_path, 'r') as f:
        reader = csv.reader(f)
        etiquetas=[]
        valores=[]
        minimos = []
        for r in reader:
            if unicos is not None and r[0] not in unicos:
                continue
            etiquetas.append(r[0])
            v = [float(i) for i in r[1:]]
            minimos.append(min(v))
            valores.append(v)

        ordenados = np.argsort(minimos)

        if n_peores>0:
            ordenados = ordenados[::-1][:n_peores]
        elif n_mejores>0:
            ordenados = ordenados[:n_mejores]

        fig = plt.figure(figsize=figsize)

        if title=='':
            plt.title('Evolución de los errores por imagen')
        else:
            plt.title(title)
        plt.xlabel('Época')
        plt.ylabel('Error')
        for indice in ordenados:
            plt.plot(valores[indice], label=etiquetas[indice])
        plt.legend()
        plt.show()



if __name__=='__main__':
    #mostrar_animacion('../corridas/NCA90/res_imgs/70')
    graficar_perdidas('../corridas/NCA90/log.csv', n_mejores=5, title='Mejores')
    for n in[4, 45,25,21,70]:
        mostrar_animacion(f'../corridas/NCA90/res_imgs/{n}')
    graficar_perdidas('../corridas/NCA90/log.csv', n_peores=5, title='Peores')
    for n in[24,12,33,48,57]:
        mostrar_animacion(f'../corridas/NCA90/res_imgs/{n}')
