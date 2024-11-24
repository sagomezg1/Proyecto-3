
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Asumiendo que tienes los datos en un archivo CSV
df = pd.read_csv('ArchivoFinal.csv')


# 2. Box plots for all score types
plt.figure(figsize=(14, 6))
score_data = data[score_columns].melt()
sns.boxplot(x='variable', y='value', data=score_data)
plt.xticks(rotation=45)
plt.title('Distribución de Puntajes por Área')
plt.xlabel('Tipo de Prueba')
plt.ylabel('Puntaje')
plt.tight_layout()
plt.show()