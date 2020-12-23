import argparse
import subprocess
import os

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
        print(f'Error: {error}')
    if mostrar: 
        print(to_print)

PWD = os.getcwd()
DIR = os.path.abspath(args.dir_base)
IMGS_DIR = os.path.join(DIR, 'imgs')
MASKS_DIR = os.path.join(DIR, 'masks')

if not os.path.isdir(DIR):
    os.makedirs(DIR)
print(f'El directorio donde se descargarán todos los datos es {DIR}')
os.chdir(DIR)

try:
    import openimages
except:
    ejecutar('pip install openimages')

print('Descarga de la base de datos')
comando = f'oi_download_dataset --base_dir {IMGS_DIR} --labels Fish '\
          f'--limit {args.n_imagenes} --format darknet'
ejecutar(comando)
print('  Base de datos descargada')

print('Obteniendo datos de segmentaciones')
ejecutar('wget https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv')
ejecutar('wget https://storage.googleapis.com/openimages/v5/train-annotations-object-segmentation.csv')
ejecutar(f'mkdir {MASKS_DIR}')
print('  Datos obtenidos')

sufixes = list(args.f_mascaras)
for s in sufixes:
    print(f'Descargando el conjunto de datos {s}')
    ejecutar(f'wget https://storage.googleapis.com/openimages/v5/train-masks/train-masks-{s}.zip')
    print('\tDescomprimiendo...')
    ejecutar(f'unzip train-masks-{s}.zip -d masks')
    print('\tBorrando...')
    ejecutar(f'rm train-masks-{s}.zip')
    print('\tDescargado... \n')

os.chdir(PWD)