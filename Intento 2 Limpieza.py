import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

data = df = pd.read_csv('Nuevo_Archivo.csv')

# 1. Eliminar filas con valores faltantes
data_cleaned = data.dropna()

# 2. Identificar columnas categóricas
categorical_cols = data_cleaned.select_dtypes(include=['object']).columns

# Aplicar Label Encoding para columnas categóricas binarias
label_encoded_cols = []
label_encoder = LabelEncoder()

for col in categorical_cols:
    if data_cleaned[col].nunique() <= 2:
        data_cleaned[col] = label_encoder.fit_transform(data_cleaned[col])
        label_encoded_cols.append(col)

# 3. Aplicar One-Hot Encoding para columnas categóricas con más de 2 categorías
data_final = pd.get_dummies(data_cleaned, columns=[col for col in categorical_cols if col not in label_encoded_cols])

# Verificar el resultado
print(data_final.info())
print(data_final.head())


# 1. Estadísticas descriptivas
numeric_stats = data_final.describe()
print("Estadísticas descriptivas:")
print(numeric_stats)


import matplotlib.pyplot as plt
import seaborn as sns

# Pregunta 1: Combinación de características socioeconómicas y escolares

plt.figure(figsize=(10, 6))
sns.barplot(
    data=data,
    x='FAMI_ESTRATOVIVIENDA',
    y='PUNT_GLOBAL',
    hue='COLE_BILINGUE',
    ci=None,
    palette='viridis'
)
plt.title('Estrato Socioeconómico y Bilingüismo vs. Puntaje Global (Promedio)')
plt.xlabel('Estrato Socioeconómico')
plt.ylabel('Puntaje Global Promedio')
plt.legend(title='Bilingüismo', labels=['No', 'Sí'])
plt.show()

# Diagramas de caja para estrato socioeconómico vs. Puntaje Global
plt.figure(figsize=(8, 5))
sns.boxplot(data=data_cleaned, x='FAMI_ESTRATOVIVIENDA', y='PUNT_GLOBAL', palette='coolwarm')
plt.title('Estrato Socioeconómico vs. Puntaje Global')
plt.xlabel('Estrato Socioeconómico')
plt.ylabel('Puntaje Global')
plt.show()

# Pregunta 2: Influencia del nivel educativo de los padres
# Boxplot para educación del padre vs. puntaje global
plt.figure(figsize=(14, 10))
sns.boxplot(data=data_cleaned, x='FAMI_EDUCACIONPADRE', y='PUNT_GLOBAL', palette='Set3')
plt.title('Nivel Educativo del Padre vs. Puntaje Global')
plt.xlabel('Nivel Educativo del Padre')
plt.ylabel('Puntaje Global')
plt.xticks(rotation=45)
plt.show()

# Diagrama de violín para educación de la madre vs. puntaje global
plt.figure(figsize=(10, 6))
sns.violinplot(data=data_cleaned, x='FAMI_EDUCACIONMADRE', y='PUNT_GLOBAL', palette='husl')
plt.title('Nivel Educativo de la Madre vs. Puntaje Global')
plt.xlabel('Nivel Educativo de la Madre')
plt.ylabel('Puntaje Global')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(10, 6))
sns.scatterplot(data=data_cleaned, x='PUNT_INGLES', y='PUNT_GLOBAL', hue='COLE_BILINGUE', palette='coolwarm', alpha=0.6)
plt.title('Relación entre Puntaje Global y Puntaje de Inglés')
plt.xlabel('Puntaje de Inglés')
plt.ylabel('Puntaje Global')
plt.legend(title='Bilingüismo')
plt.show()

# Boxplot para Puntaje Global por tipo de jornada escolar
plt.figure(figsize=(12, 6))
sns.boxplot(data=data_final, x='COLE_JORNADA_COMPLETA', y='PUNT_GLOBAL', palette='pastel')
plt.title('Comparación de Puntaje Global por Tipo de Jornada Escolar')
plt.xlabel('Jornada Completa')
plt.ylabel('Puntaje Global')
plt.show()

plt.figure(figsize=(12, 6))
sns.violinplot(data=data_final, x='FAMI_EDUCACIONPADRE_Postgrado', y='PUNT_GLOBAL', palette='muted')
plt.title('Distribución del Puntaje Global según Nivel Educativo del Padre (Postgrado)')
plt.xlabel('Nivel Educativo del Padre (Postgrado)')
plt.ylabel('Puntaje Global')
plt.show()


plt.figure(figsize=(10, 6))
sns.barplot(data=data_final, x='COLE_NATURALEZA', y='PUNT_GLOBAL', palette='viridis')
plt.title('Puntaje Global Promedio según Naturaleza del Colegio')
plt.xlabel('Naturaleza del Colegio')
plt.ylabel('Puntaje Global Promedio')
plt.show()

sns.pairplot(data_final[['PUNT_GLOBAL', 'PUNT_INGLES', 'PUNT_MATEMATICAS', 'PUNT_LECTURA_CRITICA']])
plt.suptitle('Relaciones entre Puntajes Académicos', y=1.02)
plt.show()
