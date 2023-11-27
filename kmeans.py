# Importar bibliotecas necesarias
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Suponiendo que tenemos  un DataFrame con datos de clientes
# Aquí se cargaría tus propios datos*
data = {
    'Tiempo_en_Pagina': [10, 15, 8, 5, 20, 25, 30, 12, 18, 22],
    'Compras_Realizadas': [2, 3, 1, 0, 5, 7, 8, 2, 4, 6]
}

df = pd.DataFrame(data)

# Normalizar los datos para que todas las características tengan la misma escala
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Aplicar el algoritmo de K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)

# Visualizar los resultados
plt.scatter(df['Tiempo_en_Pagina'], df['Compras_Realizadas'],
            c=df['Cluster'], cmap='viridis')
plt.title('Resultados de K-Means')
plt.xlabel('Tiempo en Página')
plt.ylabel('Compras Realizadas')
plt.show()
