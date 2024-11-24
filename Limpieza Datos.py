import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Asumiendo que tienes los datos en un archivo CSV
df = pd.read_csv('ArchivoFinal.csv')


# Crear el DataFrame
# Primero definimos las columnas
columnas = [
    'COLE_DEPTO_UBICACION', 'COLE_BILINGUE', 'COLE_CALENDARIO', 'COLE_CARACTER',
    'COLE_GENERO', 'COLE_JORNADA', 'COLE_NATURALEZA', 'FAMI_EDUCACIONMADRE',
    'FAMI_EDUCACIONPADRE', 'FAMI_ESTRATOVIVIENDA', 'FAMI_PERSONASHOGAR',
    'FAMI_TIENEAUTOMOVIL', 'FAMI_TIENECOMPUTADOR', 'FAMI_TIENEINTERNET',
    'FAMI_TIENELAVADORA', 'PUNT_GLOBAL', 'PUNT_INGLES', 'PUNT_MATEMATICAS',
    'PUNT_SOCIALES_CIUDADANAS', 'PUNT_C_NATURALES', 'PUNT_LECTURA_CRITICA'
]

def limpiar_datos(df):
    # 1. Limpieza básica
    # Reemplazar strings vacíos por NaN
    df = df.replace('', np.nan)
    
    # 2. Corregir encoding para caracteres especiales
    *
    columnas_texto = ['COLE_CARACTER', 'COLE_JORNADA', 'FAMI_EDUCACIONMADRE', 'FAMI_EDUCACIONPADRE']
    for col in columnas_texto:
        df[col] = df[col].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    
    # 3. Estandarizar valores
    # Convertir columnas binarias a 1/0
    columnas_binarias = ['COLE_BILINGUE', 'FAMI_TIENEAUTOMOVIL', 'FAMI_TIENECOMPUTADOR', 
                        'FAMI_TIENEINTERNET', 'FAMI_TIENELAVADORA']
    for col in columnas_binarias:
        df[col] = df[col].map({'Si': 1, 'No': 0, 'S': 1, 'N': 0})
    
    # 4. Manejar valores faltantes
    # Para columnas numéricas (puntajes)
    columnas_puntajes = ['PUNT_GLOBAL', 'PUNT_INGLES', 'PUNT_MATEMATICAS', 
                        'PUNT_SOCIALES_CIUDADANAS', 'PUNT_C_NATURALES', 'PUNT_LECTURA_CRITICA']
    for col in columnas_puntajes:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].mean())
    
    # 5. Label Encoding para variables categóricas ordinales
    le = LabelEncoder()
    columnas_ordinales = ['FAMI_ESTRATOVIVIENDA', 'FAMI_EDUCACIONMADRE', 'FAMI_EDUCACIONPADRE']
    for col in columnas_ordinales:
        df[col] = le.fit_transform(df[col].astype(str))
    
    # 6. One-Hot Encoding para variables categóricas nominales
    columnas_one_hot = ['COLE_CALENDARIO', 'COLE_CARACTER', 'COLE_GENERO', 
                       'COLE_JORNADA', 'COLE_NATURALEZA']
    df = pd.get_dummies(df, columns=columnas_one_hot, prefix=columnas_one_hot)
    

    return df


df_limpio = limpiar_datos(df)

print(df.head)