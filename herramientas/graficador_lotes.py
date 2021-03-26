import numpy as np
import os
import argparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
from glob import glob

from graficador_individual import extractor_datos, flatten

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type = str)
parser.add_argument('--tamano_cuadro', type = int, default=12)
parser.add_argument('--graficos', action='store_true')
parser.add_argument('--tablas', action='store_true')
args = parser.parse_args()


files_grab = sorted(glob(os.path.join(args.data_dir, 'grabación*.csv')))
files_secc = sorted(glob(os.path.join(args.data_dir, 'secciones*.txt')))

parciales = []
for grab_path, secc_path in zip(files_grab, files_secc):
    _, res = extractor_datos(grab_path, secc_path, args)
    parciales.append(res)

shape = (len(parciales[0]), len(parciales[0][0]), len(parciales[0][0][0]))

totales = [[[[] for _ in range(shape[2])] for _ in range(shape[1])] for _ in range(shape[0])]
promedios = [[[] for _ in range(shape[1])] for _ in range(shape[0])]
desv_stds = [[[] for _ in range(shape[1])] for _ in range(shape[0])]
#Se guardan los datos en totales para acceder de la siguiente manera:
# totales[P][medición][H]
# donde P = 0-9,  medición = 0-2 (Longitud,eL, eD),  H = 0-2,
for x in range(shape[0]):
    for y in range(shape[1]):
        for z in range(shape[2]):
            totales[x][y][z] += flatten([p[x][y][z] for p in parciales])
            promedios[x][z].append(np.mean(flatten([p[x][y][z] for p in parciales])))
            desv_stds[x][z].append(np.std(flatten([p[x][y][z] for p in parciales])))
            
promedios = np.array(promedios)

#print([[np.mean(l) for l in t] for t in totales[7]])
#plt.plot(totales[7][0][2])
#plt.show()


eL_min = np.min(promedios[:, :, 1])
eL_max = np.max(promedios[:, :, 1])
norm = matplotlib.colors.Normalize(vmin=eL_min, vmax = eL_max, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.cool)

fig, ax = plt.subplots()
ax.set_xlim(0,1.5)
ax.set_ylim(0,13.5)
for x in range(shape[0]):
    plt.text(0,x+3-0.2, f'P{x+1}')
    for y in range(shape[1]):
        data_y = shape[1] - y - 1
        color = mapper.to_rgba(promedios[x,data_y,1])
        pos = (y*0.5+.25, x+3)
        ax.add_patch(Ellipse(pos, .09, .8, color=color))
for y in list(range(shape[1]))[::-1]:
    data_y = shape[1] - y - 1
    plt.text(y*0.5+0.22, 13, f'H{data_y+1}')

#Dibujo las cámaras
ax.plot([0.4, 0.6], [1,1], 'k')
ax.plot([0.9, 1.1], [1,1], 'k')
plt.text(0.465, 0.3, 'C1')
plt.text(.965, 0.3, 'C2')

ax.axis('off')
cbar = plt.colorbar(mapper, ax=ax)
cbar.set_label(r'$e_L$')


#gráficos de caja y errores
fig, ax = plt.subplots(figsize=(7,4))
bp = ax.boxplot([flatten(p[0]) for p in totales],
                labels = [f'P{n+1}' for n in range(shape[0])],
                patch_artist=True,
                sym='' ##Esconder los flyers
                )

for median in bp['medians']: 
    median.set(color ='white', 
            linewidth = 1) 
ax.grid(which='both', axis = 'y')
ax.set_xlabel('Posiciones en P')
ax.set_ylabel(r'Longitud estimada ($L$)' )

#Gráfico de los errores -----------------------------------------------------------------------
errores = np.array([[np.mean(flatten(p[1])), np.mean(flatten(p[2]))] for p in totales])
fig, ax = plt.subplots(figsize=(7,4))
ax.plot(np.arange(shape[0]), errores[:,0], 'r*-', label=r'$e_L$')
ax.set_ylabel(r'Error de estimación de longitud ($e_L$)')
ax.set_xlabel('Posiciones en P')
ax.set_xticks(np.arange(shape[0]))
ax.set_xticklabels([f'P{n+1}' for n in range(shape[0])])
ax.grid(axis='y')

ax2 = ax.twinx()
ax2.plot(np.arange(shape[0]), errores[:,1], 'b*-', label=r'$e_D$')
ax2.set_ylabel(r'Error de estimación de posición ($e_D$)')
fig.legend(loc='upper center', ncol=2)
plt.show()
