import matplotlib.pyplot as plt
import networkx          as nx
import pandas            as pd
import numpy             as np

from   imblearn.over_sampling import SMOTE
from   collections            import Counter
from   igraph                 import Graph


#Caragamos los datos
data = pd.read_csv('user_rating.txt', sep='\t', header=None, names=['User A', 'User B', 'Rating', 'Date'])
data = data.drop(['Date'], axis=1)
print('\n\n · D A T A S E T   O R I G I N A L\n' + '·'*45)
print(data)


#Reducimos el numero de nodos
num_nodes = 1000
node_list = sorted(data['User A'].unique())[:num_nodes]

data = data[data['User A'].isin(node_list)]
data = data[data['User B'].isin(node_list)]

data = data.reset_index()
data = data.drop(['index'], axis=1)


#Reseteamos los IDs de los nodos
un_A = sorted(data['User A'].unique())
un_B = sorted(data['User B'].unique())

un_A.extend([element for element in un_B if element not in un_A])
un_T = sorted(un_A)


for i in range(len(un_T)):
	data['User A'] = data['User A'].replace([un_T[i]],i)
	data['User B'] = data['User B'].replace([un_T[i]],i)

print('\n\n · D A T A S E T   P R E P R O C E S A D O\n' + '·'*45)
print(data)

un_A = sorted(data['User A'].unique())
un_B = sorted(data['User B'].unique())

un_A.extend([element for element in un_B if element not in un_A])
un_T = sorted(un_A)

num_nodes = len(un_T)
print('\n Número de nodos: ' + str(num_nodes))



#Creamos la matriz de adyacencia
adj_mat = np.zeros((num_nodes, num_nodes))

for indice_fila, fila in data.iterrows():
	adj_mat[fila['User A']][fila['User B']] = fila['Rating']



#Creamos el grafo con igraph a partir de la matriz de adyacencia
G = Graph.Weighted_Adjacency(adj_mat)


# Creamos la matriz de características
feat_mat = data.copy()


# MEDIDAS A NIVEL DE GRAFO
C1 = []

C2 = []

C3 = []

C4 = []


# Recorremos cada arista del grafo
for indice_fila, fila in data.iterrows():
	a = int(fila['User A'])
	b = int(fila['User B'])

	# Borramos la arista
	G.delete_edges([(a,b)])

	# Caracteristicas
	# ~~~~~~~~~~~~~~~

	x = G.average_path_length()
	C1.append(x)

	x = G.assortativity_degree()
	C2.append(x)

	x = G.transitivity_undirected()
	C3.append(x)

	x = G.reciprocity()
	C4.append(x)

	# Reincorporamos la arista
	G.add_edges([(a,b)])


# Anyadimos a la matriz de caracteristicas
feat_mat['AvgPathLen'] = C1
feat_mat['Assort']     = C2
feat_mat['Transit']    = C3
feat_mat['Recipr']     = C4 



# MEDIDAS A NIVEL DE NODO
# TODO: Fix this!
C1A = []
C1B = []

C2A = []
C2B = []

C3A = []
C3B = []

C4A = []
C4B = []

C5A = []
C5B = []

C6A = []
C6B = []

C7A = []
C7B = []

C8A = []
C8B = []

C9A = []
C9B = []



for indice_fila, fila in data.iterrows():
	a = int(fila['User A'])
	b = int(fila['User B'])

	# Borramos la arista
	G.delete_edges([(a,b)])

	# Caracteristicas
	# ~~~~~~~~~~~~~~~
	x = G.strength()

	C1A.append(x[a])
	C1B.append(x[b])

	x = G.pagerank()

	C2A.append(x[a])
	C2B.append(x[b])

	x = G.harmonic_centrality()

	C3A.append(x[a])
	C3B.append(x[b])
	
	x = G.eigenvector_centrality()

	C4A.append(x[a])
	C4B.append(x[b])

	x = G.eccentricity()

	C5A.append(x[a])
	C5B.append(x[b])	

	x = G.coreness()

	C6A.append(x[a])
	C6B.append(x[b])

	x = G.betweenness()

	C7A.append(x[a])
	C7B.append(x[b])

	x = G.hub_score()

	C8A.append(x[a])
	C8B.append(x[b])

	x = G.authority_score()

	C9A.append(x[a])
	C9B.append(x[b])




	# Reincorporamos la arista
	G.add_edges([(a,b)])


# Anyadimos a la matriz de caracteristicas
feat_mat['StrengthA'] = C1A
feat_mat['StrengthB'] = C1B

feat_mat['PagerankA'] = C2A
feat_mat['PagerankB'] = C2B

feat_mat['HarmCenA'] = C3A
feat_mat['HarmCenB'] = C3B

feat_mat['EigenCenA'] = C4A
feat_mat['EigenCenB'] = C4B

feat_mat['EccentrA'] = C5A
feat_mat['EccentrB'] = C5B

feat_mat['CorenessA'] = C6A
feat_mat['CorenessB'] = C6B

feat_mat['BetweennA'] = C7A
feat_mat['BetweennB'] = C7B

feat_mat['HubScrA'] = C8A
feat_mat['HubScrB'] = C8B

feat_mat['AuthScrA'] = C9A
feat_mat['AuthScrB'] = C9B



# Eliminamos la informacion de los nodos
feat_mat = feat_mat.drop(['User A'], axis=1)
feat_mat = feat_mat.drop(['User B'], axis=1)


# Separamos varibles predictoras de variable a predecir para todo el conjunto de datos
X = feat_mat.drop(['Rating'], axis=1)
y = feat_mat['Rating']

# Aplicamos SMOTE para la clase minoritaria
oversample = SMOTE('minority')
newX, newy = oversample.fit_resample(X, y)


counter = Counter(y)
print('\nDistribución inicial de clases: ')
print(counter)
print('\n')
print('Distribución final de clases: ')
counter = Counter(newy)
print(counter)


newX['Rating'] = newy

print('\n' + '·'*45 + '\n· M E D I D A S   D E   C E N T R A L I D A D ·\n' + '·'*45)
print(newX)

print(newX['Rating'].value_counts())


newX.to_csv('dataset.csv', header=True, index=False)