import argparse
import subprocess
import os
from pprint import pprint as pp
from zipfile import ZipFile
from glob import glob

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
parser.add_argument('--solo', type=str,
                    help='Selecciona: mascaras, imagenes',
                    default='')
parser.add_argument('--descomprimir', type=bool,
                    help='Realizar o no la descompresion de las mascaras', default=True)
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
if not os.path.isdir(DIR):
    os.makedirs(DIR)
os.chdir(DIR)
try:
    import openimages
except:
    print('Instalando el módulo OpenImages...')
    ejecutar('pip install openimages')
if args.solo in ['imagenes','']:
    print('Descarga de la base de datos')
    comando = f'oi_download_dataset --base_dir {DIR} --labels Fish '\
              f'--limit {args.n_imagenes} --format darknet'
    ejecutar(comando)
    print('  Base de datos de las imagenes descargada')

imgs_disp = [os.path.basename(i) for i in glob('fish/images/*.*')]

if args.solo in ['mascaras','']:
    print('Obteniendo datos de segmentaciones')
    ejecutar('wget https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv')
    ejecutar('wget https://storage.googleapis.com/openimages/v5/train-annotations-object-segmentation.csv')
    ejecutar(f'mkdir masks')
    print('  Datos obtenidos')

    sufixes = args.f_mascaras
    print(f'Obteniendo los archivos {', '.join(list(sufixes))}')
    for s in sufixes:
        print(f'Descargando el conjunto de datos {s}')
        ejecutar(f'wget https://storage.googleapis.com/openimages/v5/train-masks/train-masks-{s}.zip')
        if args.descomprimir:
            print('\tDescompresión...')
            with ZipFile(f'train-masks-{s}.zip', 'r') as z:
                print('\tBuscando archivos válidos...', end=' ')
                valids = [p for p in z.namelist() if p.split('_')[0]+'.jpg' in imgs_disp]
                print(f'{len(valids)} encontrados. Descomprimiendo...')
                z.extractall('masks', valids)
                print(f'\tSe extrayeron con éxito.')
            #ejecutar(f'unzip train-masks-{s}.zip -d masks')
            print('\tBorrando...')
            ejecutar(f'rm train-masks-{s}.zip')
            print('\tTerminado :D \n')

os.chdir(PWD)
