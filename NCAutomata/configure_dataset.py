import argparse
import subprocess
import os
from pprint import pprint as pp

parser = argparse.ArgumentParser()
parser.add_argument('dir_base', type=str,
                    help='Directorio base donde se descargarán todos los archivos')
parser.add_argument('--n_imagenes', type=int,
                    help='Numero de imágenes de peces a descargar', 
                    default = 5000)
parser.add_argument('--f_mascaras', type=str,
                    help='Archivos de los datos de las máscaras a descargar, '\
                         'escribir en una sola cadena. Por defecto es 0123456789abcdef',
                    default='0123456789abcdef')
args = parser.parse_args()


def ejecutar(comando, mostrar=False):
    out = subprocess.Popen(comando.split(' '),
                       stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
    to_print, error = out.communicate()
    if error is not None: 
        pp(f'Error: {error}')
    if mostrar: 
        pp(to_print)

PWD = os.getcwd()
DIR = os.path.abspath(args.dir_base)
try:
    import openimages
except:
    ejecutar('pip install openimages')

print('Descarga de la base de datos')
comando = f'oi_download_dataset --base_dir {DIR} --labels Fish '\
          f'--limit {args.n_imagenes} --format darknet'
ejecutar(comando, mostrar=False)
print('  Base de datos descargada')

os.chdir(DIR)

print('Obteniendo datos de segmentaciones')
ejecutar('wget https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv')
ejecutar('wget https://storage.googleapis.com/openimages/v5/train-annotations-object-segmentation.csv')
ejecutar(f'mkdir masks')
print('  Datos obtenidos')

sufixes = list(args.f_mascaras)
print(f'Obteniendo los archivos {sufixes}')
for s in sufixes:
    print(f'Descargando el conjunto de datos {s}')
    ejecutar(f'wget https://storage.googleapis.com/openimages/v5/train-masks/train-masks-{s}.zip')
    print('\tDescomprimiendo...')
    ejecutar(f'unzip train-masks-{s}.zip -d masks')
    print('\tBorrando...')
    ejecutar(f'rm train-masks-{s}.zip')
    print('\tDescargado... \n')

os.chdir(PWD)
