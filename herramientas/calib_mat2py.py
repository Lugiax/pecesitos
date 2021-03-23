import numpy as np
import os
import argparse
from scipy.io import loadmat
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('filepath', type = str)
parser.add_argument('--save_dir', type = str, default='')

args = parser.parse_args()

carpeta = os.path.split(args.filepath)[0]
save_dir = args.save_dir if args.save_dir!='' else carpeta

##Lectura de los datos
m = loadmat(os.path.join(carpeta, 'calibData.mat'))
m['distCoeffs1'] =np.array([m['distCoeffs1'].tolist()[0] + [0,0,0]])
m['distCoeffs2'] =np.array([m['distCoeffs2'].tolist()[0] + [0,0,0]])
m['F'] /= m['F'][2,2]
m['tamano_cuadro'] = m['tamano_cuadro'][0][0]

vals = ['E', 'F', 'R', 'T', 'cameraMatrix1', 'cameraMatrix2', 'distCoeffs1', 'distCoeffs2', 'tamano_cuadro']

data = {k: m[k] for k in vals}

with open(os.path.join(save_dir, 'calibData_frommat.pk'), 'wb') as f:
    pickle.dump(data, f)