import matplotlib
#matplotlib.use('Agg')
import scipy.io
import re
import os
import time
from hdas_features_fft_frec_sp import HDAS

# Registra el tiempo de inicio
start_time = time.time()

# filenames = ['2021_11_26_00h38m30s_HDAS_2Dmap_Strain.bin',
#              '2021_11_26_00h39m30s_HDAS_2Dmap_Strain.bin',
#              '2021_11_26_00h40m30s_HDAS_2Dmap_Strain.bin',
#              '2021_11_26_00h41m30s_HDAS_2Dmap_Strain.bin',
#              '2021_11_26_00h42m30s_HDAS_2Dmap_Strain.bin',
#              '2021_11_26_00h43m30s_HDAS_2Dmap_Strain.bin']

# filenames = ['2021_11_26_00h46m30s_HDAS_2Dmap_Strain.bin',
#               '2021_11_26_00h47m30s_HDAS_2Dmap_Strain.bin',
#               '2021_11_26_00h48m30s_HDAS_2Dmap_Strain.bin',
#               '2021_11_26_00h49m30s_HDAS_2Dmap_Strain.bin',
#               '2021_11_26_00h50m30s_HDAS_2Dmap_Strain.bin',
#               '2021_11_26_00h51m30s_HDAS_2Dmap_Strain.bin']

filenames = ['2021_11_26_02h31m30s_HDAS_2Dmap_Strain.bin',
              '2021_11_26_02h32m30s_HDAS_2Dmap_Strain.bin',
              '2021_11_26_02h33m30s_HDAS_2Dmap_Strain.bin',
              '2021_11_26_02h34m30s_HDAS_2Dmap_Strain.bin',
              '2021_11_26_02h35m30s_HDAS_2Dmap_Strain.bin',
              '2021_11_26_02h36m30s_HDAS_2Dmap_Strain.bin']

# Lista para almacenar los archivos existentes
existing_files_names = []

# Lista para almacenar los archivos existentes (con 0 y 1)
existing_files = []

# Crea una instancia de HDAS para cada archivo y verifica su existencia
for filename in filenames:
    if os.path.exists(os.path.join('C:\InfoTFG', filename)):
        existing_files_names.append(filename)
        existing_files.append('1')  # Agrega '1' si el archivo existe
   
    else:
        existing_files.append('0')  # Agrega '0' si el archivo no existe

# Ahora, existing_files_names contiene solo los nombres de los archivos que existen
print("Archivos existentes:", existing_files_names)

# Imprimir los archivos que no existen
non_existing_files = [filename for filename in filenames if filename not in existing_files_names]
print("Archivos no existentes:", non_existing_files)


# Usa existing_files_names para crear una instancia de HDAS
H = HDAS('C:\InfoTFG', existing_files_names)

H.check()

# print("DECIMACION")
# H.decimateX(1/5, dectype='simple')

#print("COH")
#H.removeCoherentNoise(method='fit')

print("DETREND")
start_proc = time.time()  
H.removeTrend()

print("COH")
H.removeCoherentNoise(method='fit')

print("FILTER")
H.filter(1.5,20.0)
end_proc = time.time()
#H.filter(0.4,0.6)

execution_proc = end_proc - start_proc

print(f"Tiempo total de pre-procesado: {execution_proc:.2f} segundos")


#print("DECIMATE")
#H.decimateT(10)

#print("COH")
#H.removeCoherentNoise(method='fit')

# print("CUT")
# w=10
# H.cutT(w,H.len())

# print("CUTX")
# w=10
# H.cutX(0, 2000)


# print("NORM")
# H.normalize()

#print("DECX")
#H.decimateX(5,'median')


# Especifica los puntos espaciales inicial y final que deseas seleccionar
inicio_punto_espacial = 0  # Punto espacial inicial 
fin_punto_espacial = 200  # Punto espacial final (exclusivo)


# Itera sobre los puntos espaciales seleccionados
for i in range(inicio_punto_espacial, fin_punto_espacial):
    
    # Registra el tiempo de inicio para el punto espacial actual
    start_time_sp = time.time()

    H.fft_128bin(sp=i)

    
    # Calcula el tiempo de ejecución del punto espacial actual
    execution_time_sp = time.time() - start_time_sp
    
    # Muestra el tiempo de ejecución del punto espacial actual
    print(f"Tiempo de ejecución del punto espacial {i}: {execution_time_sp:.2f} segundos")
    
    
    # Extrae la parte de la fecha y hora del primer y último nombre de archivo
    first_datetime_match = re.search(r'\d{4}_\d{2}_\d{2}_\d{2}h\d{2}m\d{2}s', filenames[0])
    last_datetime_match = re.search(r'\d{4}_\d{2}_\d{2}_\d{2}h\d{2}m\d{2}s', filenames[-1])
    
    # Extrae las partes de la fecha y hora individualmente
    first_datetime = first_datetime_match.group()
    last_datetime = last_datetime_match.group()
    
    # Elimina los guiones bajos y caracteres no deseados de las cadenas de fecha y hora
    first_datetime_clean = re.sub(r'_|h|m|s', '', first_datetime)
    last_datetime_clean = re.sub(r'_|h|m|s', '', last_datetime)
    
    
    # Obtén el número de iteración con formato 0000-4959
    iteration_number = str(i).zfill(4)  # Asegura que el número tenga al menos 4 dígitos
    
    # Construye el nombre del archivo .mat con el número de iteración
    output_filename = f"SF_{first_datetime_clean[:8]}_{first_datetime_clean[8:]}_{last_datetime_clean[8:]}_SP{iteration_number}_20.mat"

    # Construye el nombre del diccionario con el número de iteración
        
    features_name = f"SF_{first_datetime_clean[:8]}_{first_datetime_clean[8:]}_{last_datetime_clean[8:]}_SP{iteration_number}_20"



    
    # Crea un diccionario para almacenar las matrices de datos
    features = {
        features_name: {
            'FFT_128_BIN': H.val_for_fbin,
            'FFT_128_BIN_delta_1': H.deltas_fft,
            'FFT_128_BIN_delta_2': H.deltas_deltas_fft,
            
            
            'Existing_files': ''.join(existing_files)
        }
    }
    
    
    # Guarda el diccionario en un archivo .mat
    scipy.io.savemat(output_filename, features)
    
    
    
    # # Guarda el diccionario en un archivo .mat
    # scipy.io.savemat(output_filename, {'features': features})



# Registra el tiempo de finalización
end_time = time.time()

# Calcula el tiempo total de ejecución en segundos
execution_time_seconds = end_time - start_time

# Convierte el tiempo total de ejecución a horas, minutos y segundos
hours = int(execution_time_seconds // 3600)
minutes = int((execution_time_seconds % 3600) // 60)
seconds = int(execution_time_seconds % 60)

# Imprime el tiempo de ejecución
print("Tiempo de ejecución total:", hours, "horas,", minutes, "minutos y", seconds, "segundos")
