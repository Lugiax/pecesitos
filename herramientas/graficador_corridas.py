import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import norm
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type = str)
parser.add_argument('tipo_pez', type = str)
parser.add_argument('--tamano_cuadro', type = int, default=12)

args = parser.parse_args()

csv_path = os.path.join(args.data_dir, 'grabación.csv')
secciones_path = os.path.join(args.data_dir, 'secciones.txt')
pez = args.tipo_pez

## Se lee el archivo de datos
#El csv debe tener el siguiente formato:
#  frame,p1x,p1y,p1z,p2x,p2y,p2z,distancia_inmediata,angulo_inmediato,distanacia_RANSAC,angulo_prom

data = pd.read_csv(csv_path, header=0)

data ['profundidad'] = (np.sqrt(data.p1x**2 + data.p1y**2 + data.p1z**2)+
                       np.sqrt(data.p2x**2 + data.p2y**2 + data.p2z**2))*args.tamano_cuadro/2

#lectura de las secciones:
# El archivo secciones debe tener el siguiente formato:
#  PX , HX : inicio - final
#Donde inicio y final se refiere los números correspondientes
#a los frames donde comienza y termina la sección, y PX , HX
#son las coordenadas de la posición
if not os.path.isfile(secciones_path):
    plt.plot(data.frame, data.profundidad,'.')
    plt.show()
    #Se procede a ingresar los datos
    print('Ingresar los límites en la forma L1 - L2')
    Ps, Hs = (1,2,3,4,5,6), (1,2,3)
    lines=[]
    for H in Hs:
        for P in Ps:
            coord = f'P{P} , H{H}'
            res = input(f'{coord} > ')
            lines.append(f'{coord} : {res}')
    with open(secciones_path, 'w') as f:
        f.writelines('\n'.join(lines))
    print('Archivo creado')

secciones = []
with open(secciones_path, 'r') as f:
    lineas = f.readlines()
    for l in li'888888888888888888
































































































































































































































































































































































    666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666635555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555--------------------------------------------------333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333
        if 'NA' in inic_fin:
            secciones.append([-1, -1])
        else:
            secciones.append([int(x.strip()) for x in inic_fin.split('-')])


correcciones_z = {'piraña': -12,
                'cirujano': -18}
longitudes = {'piraña': 66,
                'cirujano': 70}
centro_cam = [95+80,160,55]

posiciones_p = [195,245,295,345,395,445]
posiciones_h = [170-86, 207-86, 257-86]
altura = 160

# Hs en columnas Ps en filas
coordenadas_pos = np.array([[[ph - centro_cam[0],
                              altura - centro_cam[1],
                              pp+correcciones_z[pez] - centro_cam[2]]\
                            for ph in posiciones_h]\
                            for pp in posiciones_p])
profundidades = norm(coordenadas_pos, axis=2)
#print(profundidades)

calculos = []
por_profundidades = []
datos_para_tabla_estimados = ''
datos_para_tabla_reales = ''
for i, P_z in enumerate(profundidades):
    calculos.append([])
    por_profundidades.append([[],[],[],[]])
    datos_para_tabla_estimados += f'P{i+1}&'
    datos_para_tabla_reales += f'P{i+1}&'
    for j, hx in enumerate(P_z):
        inicio, fin = secciones[i+j*6]
        corrida = data[data.frame.between(inicio, fin, inclusive=True)]
        dist = corrida.distancia_inmediata
        prof = corrida.profundidad
        e_l = ( 1 - corrida.distancia_inmediata / longitudes[pez])**2 
        e_p = ( 1 - corrida.profundidad / hx)**2

        calculos[i].append([np.mean(dist), np.std(dist), np.mean(prof), np.std(prof), np.mean(e_l), np.mean(e_p)])
        por_profundidades[i][0] += dist.tolist()
        por_profundidades[i][1] += prof.tolist()
        por_profundidades[i][2] += e_l.tolist()
        por_profundidades[i][3] += e_p.tolist()

        datos_para_tabla_estimados += '{\\scriptsize %.2f}$\\pm${\\tiny %.2f}&{\\scriptsize %.2f}$\\pm${\\tiny %.2f}&'\
                               %(calculos[i][j][0],   calculos[i][j][1],  calculos[i][j][2],  calculos[i][j][3])
        datos_para_tabla_reales += '%.2f&'%(hx)
    datos_para_tabla_estimados = datos_para_tabla_estimados[:-1] + '\\\\'
    datos_para_tabla_reales = datos_para_tabla_reales[:-1] + '\\\\'


print('Copia para tabla de datos de profundidades ---------------------------------------')
datos_para_tabla_estimados = """\\begin{table}[h!]
    \\centering
    \\begin{tabular}{|c||c|c|c|c|c|c|}
    \\hline
    \\multirow{2}{*}{} & \\multicolumn{2}{c|}{H1} & \\multicolumn{2}{c|}{H2} & \\multicolumn{2}{c|}{H3} \\\\
        & $L$ & $P_Z$ & $L$ & $P_Z$ & $L$ & $P_Z$\\\\ 
        \\hline\\hline 
        """ + datos_para_tabla_estimados.replace('nan','SD').replace('$\\pm${\\tiny SD}','') +"""
    \\end{tabular}
    \\caption{Resultados de las mediciones de prueba con el \\textit{phantom} de pez LLENAR. Todas las unidades son mm. Por cada posición (P,H) se dan dos mediciones: la longitud ($L$) y la profundidad ($P_Z$) estimadas por el sistema. La longitud real de la figura es de 70mm.}
    \\label{tab:resultados_profundidad_LLENAR}
\\end{table} %Obtenido de herramientas/graficador_corridas.py :D
"""
print(datos_para_tabla_estimados)
print('--------------------------------------------------------------------------')

print('\nCopia para tabla de datos reales de profundidades ---------------------------------------')
print(datos_para_tabla_reales+' %Obtenido de herramientas/graficador_corridas.py :D')
print('--------------------------------------------------------------------------')

calculos = np.array(calculos)

#Gráfico de longitudes por profundidades-------------------------------------------------------------------------------------------
colors = ['#1D3461', '#1F487E',  
          '#376996', '#6290C8',
          '#7296C2', '#829CBC']
fig, ax = plt.subplots(figsize=(7,4))
bp = ax.boxplot([p[0] for p in por_profundidades],
                labels = [f'P{n+1}' for n in range(len(por_profundidades))],
                patch_artist=True,
                sym='' ##Esconder los flyers
                )
for patch, color in zip(bp['boxes'], colors[::-1]): 
    patch.set_facecolor(color) 
for median in bp['medians']: 
    median.set(color ='white', 
               linewidth = 1) 
ax.grid(which='both', axis = 'y')
ax.set_xlabel('Profundidades')
ax.set_ylabel(r'Longitud estimada ($L$)' )
#plt.show()


#Gráfico de los errores -----------------------------------------------------------------------
errores = -np.log(np.array([[np.mean(p[2]), np.mean(p[3])] for p in por_profundidades]))

fig, ax = plt.subplots(figsize=(7,4))
ax.plot(errores[:,0], 'r-', label=r'$e_L$')
ax.plot(errores[:,1], 'b-', label=r'$e_Pz$')
ax.set_ylabel(r'-$\log(e)$')
ax.set_xlabel('Profundidades')
ax.set_xticklabels(['0']+[f'P{n+1}' for n in range(len(por_profundidades))])
ax.grid(axis='y')
ax.legend()
plt.show()

