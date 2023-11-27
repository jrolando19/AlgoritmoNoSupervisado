import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Creamos un DataFrame de ejemplo con datos ficticios de clientes
np.random.seed(42)
clientes = {
    'Edad': np.random.randint(18, 70, size=100),
    'Ingresos_anuales': np.random.randint(20000, 150000, size=100),
    'Gasto_anual': np.random.randint(100, 3000, size=100)
}

df = pd.DataFrame(clientes)

# Visualizamos los datos antes de la segmentación
plt.scatter(df['Ingresos_anuales'], df['Gasto_anual'])
plt.xlabel('Ingresos Anuales')
plt.ylabel('Gasto Anual')
plt.title('Datos de Clientes antes de la Segmentación')
plt.show()

# Aplicamos K-Means para segmentar a los clientes en 3 grupos
kmeans = KMeans(n_clusters=3)
kmeans.fit(df[['Ingresos_anuales', 'Gasto_anual']])

# Obtenemos las etiquetas de cluster asignadas a cada cliente
labels = kmeans.labels_

# Añadimos las etiquetas al DataFrame original
df['Cluster'] = labels

# Visualizamos los clusters después de la segmentación
plt.scatter(df['Ingresos_anuales'], df['Gasto_anual'],
            c=df['Cluster'], cmap='viridis')
plt.xlabel('Ingresos Anuales')
plt.ylabel('Gasto Anual')
plt.title('Segmentación de Clientes en 3 Grupos')
plt.show()
