import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Creamos un DataFrame de ejemplo con datos ficticios de clientes
np.random.seed(50)
clientes = {
    'Edad': np.random.randint(18, 70, size=100),
    'Ingresos_anuales': np.random.randint(20000, 150000, size=100),
    'Gasto_anual': np.random.randint(100, 3000, size=100)
}

df = pd.DataFrame(clientes)

# VISUALIZACIÓN DE DATOS EN GENERAL
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].scatter(df['Ingresos_anuales'], df['Gasto_anual'])
ax[0].set_xlabel('Ingresos Anuales')
ax[0].set_ylabel('Gasto Anual')
ax[0].set_title('Segmentación de Clientes (Ingresos vs Gasto)')

ax[1].scatter(df['Edad'], df['Ingresos_anuales'])
ax[1].set_xlabel('Edad')
ax[1].set_ylabel('Ingresos Anuales')
ax[1].set_title('Segmentación de Clientes (Edad vs Ingresos)')
plt.show()

# K-Means: Segmentación de Clientes en 3 grupos
kmeans = KMeans(n_clusters=3)
kmeans.fit(df[['Ingresos_anuales', 'Gasto_anual', 'Edad']])

# Obtenciónde etiqueta designada a cada cliente
labels = kmeans.labels_

# Añadir etiquetas al DataFrame con la etiqueta 'Cluster'
df['Cluster'] = labels

# Visualización con clasificación
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].scatter(df['Ingresos_anuales'], df['Gasto_anual'],
              c=df['Cluster'], cmap='viridis')
ax[0].set_xlabel('Ingresos Anuales')
ax[0].set_ylabel('Gasto Anual')
ax[0].set_title('Segmentación de Clientes (Ingresos vs Gasto)')

ax[1].scatter(df['Edad'], df['Ingresos_anuales'],
              c=df['Cluster'], cmap='viridis')
ax[1].set_xlabel('Edad')
ax[1].set_ylabel('Ingresos Anuales')
ax[1].set_title('Segmentación de Clientes (Edad vs Ingresos)')

plt.show()
