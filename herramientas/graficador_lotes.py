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

if args.tablas:
    tablas = True
    args.tablas = False
else:
    tablas = False
if args.graficos:
    graficos = True
    args.graficos = False
else:
    graficos=False
files_grab = sorted(glob(os.path.join(args.data_dir, 'grabación*.csv')))
files_secc = sorted(glob(os.path.join(args.data_dir, 'secciones*.txt')))
#print(files_grab)

parciales = []
for grab_path, secc_path in zip(files_grab, files_secc):
    print(f'Extrayendo datos de {grab_path}\n\t {secc_path}')
    _, res = extractor_datos(grab_path, secc_path, args)
    parciales.append(res)

shape = (len(parciales[0]), len(parciales[0][0]), len(parciales[0][0][0]))
#print(shape)
totales = [[[[] for _ in range(shape[2])] for _ in range(shape[1])] for _ in range(shape[0])]
promedios = [[[] for _ in range(shape[1])] for _ in range(shape[0])]
desv_stds = [[[] for _ in range(shape[1])] for _ in range(shape[0])]
datos_para_tabla_estimados = ''
#Se guardan los datos en totales para acceder de la siguiente manera:
# totales[P][medición][H]
# donde P = 0-9,  medición = 0-2 (Longitud,eL, eD),  H = 0-2,
for x in range(shape[0]):
    datos_para_tabla_estimados += f'P{x+1}&'
    for y in range(shape[1]):
        for z in range(shape[2]):
            todos = flatten([p[x][y][-(z+1)] for p in parciales])
            totales[x][y][z] += todos
            promedio = np.mean(todos)
            desv_std = np.std(todos)
            promedios[x][z].append(promedio)
            
            if y==0:
                datos_para_tabla_estimados += '{\\small %.2f}$\\pm${\scriptsize %.2f}&'\
                                %(promedio, desv_std)
                if graficos:
                    fig, ax = plt.subplots()
                    for i, p in enumerate(parciales):
                        ax.set_title(f'P{x+1} - H{z+1}')
                        ax.plot(p[x][y][z], label=f'corrida {i+1}')
                    plt.legend()
                    plt.show()
    datos_para_tabla_estimados = datos_para_tabla_estimados[:-1] + '\\\\ \n'
                
            
                        
promedios = np.array(promedios)

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

if tablas:
    print('\nCopia para tabla de datos de profundidades ---------------------------------------')
    datos_para_tabla_estimados = """\\begin{table}[h!]
    \\centering
    \\begin{tabular}{c||c|c|c|}
    & H3 & H2 & H1 \\\\
    \\hline\\hline 
    """ + datos_para_tabla_estimados.replace('nan','SD').replace('$\\pm${\\scriptsize SD}','') +\
    '\\hline \\end{tabular}'+\
    '\\caption{Resultados de las mediciones de prueba con el \\textit{phantom} de pez LLENAR. Todas las unidades están dadas en milímetros. Por cada posición $(H,P)$ se muestra la longitud estimada promedio y la desviación estándar de las mediciones realizadas. '+\
    'La longitud real de la figura es de LLENARmm.}\\label{tab:resultados_LLENAR} \\end{table} %Obtenido de herramientas/graficador_corridas.py :D'
    print(datos_para_tabla_estimados)
    print('--------------------------------------------------------------------------')
