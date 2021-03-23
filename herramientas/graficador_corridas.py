import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import norm
from scipy.spatial.distance import euclidean
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type = str)
parser.add_argument('tipo_pez', type = str)
parser.add_argument('--tamano_cuadro', type = int, default=12)
parser.add_argument('--mostrar_distancias', action='store_true')

args = parser.parse_args()

pez = args.tipo_pez
correcciones_z = {'piraña': -12,
                'cirujano': -18}
longitudes = {'piraña': 66,
                'cirujano': 70}


altura = 305-80
posiciones_p = [i*10+95-correcciones_z[pez] for i in list(range(16,35,2))]
posiciones_h = [170-100, 207-100, 257-100]
centro_cam = [180,altura,55]

n_Ps, n_Hs = len(posiciones_p), len(posiciones_h)

csv_path = os.path.join(args.data_dir, 'grabación.csv')
secciones_path = os.path.join(args.data_dir, 'secciones.txt')

## Se lee el archivo de datos
#El csv debe tener el siguiente formato:
#  frame,p1x,p1y,p1z,p2x,p2y,p2z,distancia_inmediata,angulo_inmediato,distanacia_RANSAC,angulo_prom

data = pd.read_csv(csv_path, header=0)

#Se estima la profundidad para extraer fácilmente los intervalos
data ['profundidad'] = (np.sqrt(data.p1x**2 + data.p1y**2 + data.p1z**2)+
                       np.sqrt(data.p2x**2 + data.p2y**2 + data.p2z**2))*args.tamano_cuadro/2

#lectura de las secciones:
# El archivo secciones debe tener el siguiente formato:
#  PX , HX : inicio - final
#Donde inicio y final se refiere los números correspondientes
#a los frames donde comienza y termina la sección, y PX , HX
#son las coordenadas de la posición
if not os.path.isfile(secciones_path) or args.mostrar_distancias:
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.plot(data.frame, data.profundidad, 'r-')
    ax.set_ylabel('Profundidad estimada')
    ax2.plot(data.frame, data.distancia_inmediata, 'b-')
    ax2.set_ylabel('Longitud estimada')
    plt.show()

if not os.path.isfile(secciones_path):
    #Se procede a ingresar los datos
    print('Ingresar los límites en la forma L1 - L2')
    lines=[]
    for H in range(n_Hs):
        for P in range(n_Ps):
            coord = f'P{P+1} , H{H+1}'
            res = input(f'{coord} > ')
            lines.append(f'{coord} : {res}')
    with open(secciones_path, 'w') as f:
        f.writelines('\n'.join(lines))
    print('Archivo creado')

secciones = []
with open(secciones_path, 'r') as f:
    lineas = f.readlines()
    for l in lineas:
        _, inic_fin = l.split(':')
        if 'NA' in inic_fin:
            secciones.append([-1, -1])
        else:
            secciones.append([int(x.strip()) for x in inic_fin.split('-')])


# Hs en columnas Ps en filas
coordenadas_pos = np.array([[[centro_cam[0] - ph, #Por la inversión de ejes
                              altura - centro_cam[1],
                              pp - centro_cam[2]]\
                            for ph in posiciones_h]\
                            for pp in posiciones_p])
#profundidades = norm(coordenadas_pos, axis=2)
#print(profundidades)

calculos = []
por_profundidades = []
datos_para_tabla_estimados = ''
#datos_para_tabla_reales = ''
for i, fila in enumerate(coordenadas_pos):
    calculos.append([])
    por_profundidades.append([[],[],[]])
    datos_para_tabla_estimados += f'P{i+1}&'
    #datos_para_tabla_reales += f'P{i+1}&'
    for j, coord in enumerate(fila):
        inicio, fin = secciones[i+j*n_Ps]
        corrida = data[data.frame.between(inicio, fin, inclusive=True)]
        print(f'\n P{i+1}-H{j+1} : {inicio} - {fin} --- {corrida.shape}')
        dist = corrida.distancia_inmediata
        prof = corrida.profundidad
        e_l = ( 1 - corrida.distancia_inmediata / longitudes[pez])**2 
        
        
        coord_estimada = (np.array([corrida.p1x, corrida.p1y, corrida.p1z]).T + 
                       np.array([corrida.p2x, corrida.p2y, corrida.p2z]).T)*args.tamano_cuadro/2
        coord_real = np.repeat([coord], coord_estimada.shape[0], axis=0)
        e_d = np.sqrt( (coord_estimada[:,0] - coord_real[:,0])**2 + 
                       (coord_estimada[:,1] - coord_real[:,1])**2 +
                       (coord_estimada[:,2] - coord_real[:,2])**2 )
        

        calculos[i].append([np.mean(dist), np.std(dist), np.mean(e_l), np.std(e_l), np.mean(e_d), np.std(e_d)])
        por_profundidades[i][0] += dist.tolist()
        por_profundidades[i][1] += e_l.tolist()
        por_profundidades[i][2] += e_d.tolist()

        datos_para_tabla_estimados += '{\\scriptsize %.2f}$\\pm${\\tiny %.2f}&'\
                               %(calculos[i][j][0],   calculos[i][j][1])
        #datos_para_tabla_reales += '%.2f&'%(hx)
    datos_para_tabla_estimados = datos_para_tabla_estimados[:-1] + '\\\\ \n'
    #datos_para_tabla_reales = datos_para_tabla_reales[:-1] + '\\\\'


print('\nCopia para tabla de datos de profundidades ---------------------------------------')
datos_para_tabla_estimados = """\\begin{table}[h!]
\\centering
\\begin{tabular}{c||c|c|c|}
& H1 & H2 & H3 \\\\
\\hline\\hline 
""" + datos_para_tabla_estimados.replace('nan','SD').replace('$\\pm${\\tiny SD}','') +\
'\\hline \\end{tabular}'+\
'\\caption{Resultados de las mediciones de prueba con el \\textit{phantom} de pez %s. '%pez+\
'Todas las unidades son mm. Por cada posición (P,H) se muestra la longitud estimada promedio y la desviación estándar de las mediciones realizadas. '+\
'La longitud real de la figura es de %dmm.}'%longitudes[pez]+\
'\\label{tab:resultados_profundidad_%s}'%pez+\
'\\end{table} %Obtenido de herramientas/graficador_corridas.py :D'
print(datos_para_tabla_estimados)
print('--------------------------------------------------------------------------')

#print('\nCopia para tabla de datos reales de profundidades ---------------------------------------')
#print(datos_para_tabla_reales+' %Obtenido de herramientas/graficador_corridas.py :D')
#print('--------------------------------------------------------------------------')

calculos = np.array(calculos)

#Gráfico de longitudes por profundidades-------------------------------------------------------------------------------------------
#colors = ['#1D3461', '#1F487E',  
#          '#376996', '#6290C8',
#          '#7296C2', '#829CBC']
fig, ax = plt.subplots(figsize=(7,4))
bp = ax.boxplot([p[0] for p in por_profundidades],
                labels = [f'P{n+1}' for n in range(len(por_profundidades))],
                patch_artist=True,
                sym='' ##Esconder los flyers
                )
#for patch, color in zip(bp['boxes'], colors[::-1]): 
#    patch.set_facecolor(color) 
for median in bp['medians']: 
    median.set(color ='white', 
               linewidth = 1) 
ax.grid(which='both', axis = 'y')
ax.set_xlabel('Posiciones en P')
ax.set_ylabel(r'Longitud estimada ($L$)' )
#plt.show()


#Gráfico de los errores -----------------------------------------------------------------------
errores = np.array([[np.mean(p[1]), np.mean(p[2])] for p in por_profundidades])
fig, ax = plt.subplots(figsize=(7,4))
ax.plot(np.arange(n_Ps), errores[:,0], 'r-', label=r'$e_L$')
ax.set_ylabel(r'Error de estimación de longitud ($e_L$)')
ax.set_xlabel('Posiciones P')
ax.set_xticks(np.arange(n_Ps))
ax.set_xticklabels([f'P{n+1}' for n in range(n_Ps)])
ax.grid(axis='y')

ax2 = ax.twinx()
ax2.plot(np.arange(n_Ps), errores[:,1], 'b-', label=r'$e_D$')
ax2.set_ylabel(r'Error de estimación de posición ($e_D$)')
fig.legend(loc='upper center', ncol=2)
plt.show()

