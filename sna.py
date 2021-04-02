import matplotlib.pyplot as plt
import networkx          as nx
import pandas            as pd
import numpy             as np

from   imblearn.over_sampling import SMOTE
from   collections            import Counter
from   igraph                 import Graph


def info(data):
	print('\n' + '·'*50 + '\nData head:')
	print(data)
	print('Dimensionalidad de la tabla: ' + str(data.shape[0]) + 'x' + str(data.shape[1]))


#Caragamos los datos
data = pd.read_csv('user_rating.txt', sep='\t', header=None, names=['User A', 'User B', 'Rating', 'Date'])
data = data.drop(['Date'], axis=1)
info(data)


#Reducimos el numero de nodos
num_nodes = 500
node_list = sorted(data['User A'].unique())[:num_nodes]
node_list_rep = range(len(node_list))
print('\nNos quedamos con ' + str(len(node_list)) + ' nodos')

data = data[data['User A'].isin(node_list)]
data = data[data['User B'].isin(node_list)]

data = data.reset_index()
data = data.drop(['index'], axis=1)

#info(data)


#Reseteamos los IDs de los nodos
un_A = sorted(data['User A'].unique())
un_B = sorted(data['User B'].unique())

un_A.extend([element for element in un_B if element not in un_A])
un_T = sorted(un_A)


for i in range(len(un_T)):
	# print(' Remplazamos ' + str(un_T[i]) + ' por ' + str(i))
	data['User A'] = data['User A'].replace([un_T[i]],i)
	data['User B'] = data['User B'].replace([un_T[i]],i)

info(data)

un_A = sorted(data['User A'].unique())
un_B = sorted(data['User B'].unique())

un_A.extend([element for element in un_B if element not in un_A])
un_T = sorted(un_A)

num_nodes = len(un_T)
print('\nNúmero de nodos ' + str(num_nodes))


#Matriz de adyacencia
print('\n' + '·'*50 + '\nCreamos la matriz de adyacencia...')
adj_mat = np.zeros((num_nodes, num_nodes))

for indice_fila, fila in data.iterrows():
	adj_mat[fila['User A']][fila['User B']] = fila['Rating']

#print(adj_mat)

feat_mat = data.copy()
print('\n' + '·'*50 + '\nMatriz de caracteristicas')
print(feat_mat)


print('\n' + '·'*50 + '\nCreamos el grafo con Igraph...')
G = Graph.Weighted_Adjacency(adj_mat)
#print(G.summary(verbosity=1))


# MEDIDAS A NIVEL DE GRAFO
# Av path lenght
C1 = []

# assortativity_degree
C2 = []

# transitivity_undirected
C3 = []

# transitivity_avglocal_undirected
C4 = []

# reciprocity
C5 = []


for indice_fila, fila in data.iterrows():
	a = int(fila['User A'])
	b = int(fila['User B'])

	print('\n Boramos ' + str(a) + '->' + str(b))
	G.delete_edges([(a,b)])

	# Caracteristicas
	# ~~~~~~~~~~~~~~~

	# Av path lenght
	x = G.average_path_length()
	C1.append(x)
	print(' > average_path_length: ' + str(x))

	# assortativity_degree
	x = G.assortativity_degree()
	C2.append(x)
	print(' > assortativity_degree: ' + str(x))

	# transitivity_undirected
	x = G.transitivity_undirected()
	C3.append(x)
	print(' > transitivity_undirected: ' + str(x))

	# transitivity_avglocal_undirected
	x = G.transitivity_avglocal_undirected()
	C4.append(x)
	print(' > transitivity_avglocal_undirected: ' + str(x))

	# reciprocity
	x = G.reciprocity()
	C5.append(x)
	print(' > reciprocity: ' + str(x))

	print('\n Anyadimos ' + str(a) + '->' + str(b))
	G.add_edges([(a,b)])

	print('¨' * 60)


feat_mat['G.Av path']    = C1
feat_mat['G.Assort d'] = C2
feat_mat['G.Trans']      = C3
feat_mat['G.Trans Avg']   = C4
feat_mat['G.Reciprocity'] = C5 


'''
# MEDIDAS A NIVEL DE ARISTA
#Esto probablemente es hacer trampa!!
feat_mat['E.Mutual'] = G.is_mutual()
feat_mat['E.Between'] = G.edge_betweenness()

print(feat_mat)
'''


# MEDIDAS A NIVEL DE NODO
# strength
C1A = []
C1B = []

# pagerank
C2A = []
C2B = []

# harmonic_centrality
C3A = []
C3B = []

# eigenvector_centrality
C4A = []
C4B = []

# eccentricity
C5A = []
C5B = []

# coreness
C6A = []
C6B = []

# betweeness
C7A = []
C7B = []

# hub_score
C8A = []
C8B = []

# authority_score
C9A = []
C9B = []



for indice_fila, fila in data.iterrows():
	a = int(fila['User A'])
	b = int(fila['User B'])

	print('\n Boramos ' + str(a) + '->' + str(b))
	G.delete_edges([(a,b)])

	# Caracteristicas
	# ~~~~~~~~~~~~~~~

	# strength
	print(' > strength')
	x = G.strength()

	print('  A: ' + str(x[a]))
	print('  B: ' + str(x[b]))

	C1A.append(x[a])
	C1B.append(x[b])


	# pagerank
	print(' > pagerank')
	x = G.pagerank()

	print('  A: ' + str(x[a]))
	print('  B: ' + str(x[b]))

	C2A.append(x[a])
	C2B.append(x[b])


	# harmonic_centrality
	print(' > harmonic_centrality')
	x = G.harmonic_centrality()

	print('  A: ' + str(x[a]))
	print('  B: ' + str(x[b]))

	C3A.append(x[a])
	C3B.append(x[b])
	

	# eigenvector_centrality
	print(' > eigenvector_centrality')
	x = G.eigenvector_centrality()

	print('  A: ' + str(x[a]))
	print('  B: ' + str(x[b]))

	C4A.append(x[a])
	C4B.append(x[b])


	# eccentricity
	print(' > eccentricity')
	x = G.eccentricity()

	print('  A: ' + str(x[a]))
	print('  B: ' + str(x[b]))

	C5A.append(x[a])
	C5B.append(x[b])	


	# coreness
	print(' > coreness')
	x = G.coreness()

	print('  A: ' + str(x[a]))
	print('  B: ' + str(x[b]))

	C6A.append(x[a])
	C6B.append(x[b])


	# betweenness
	print(' > betweenness')
	x = G.betweenness()

	print('  A: ' + str(x[a]))
	print('  B: ' + str(x[b]))

	C7A.append(x[a])
	C7B.append(x[b])


	# hub_score
	print(' > hub_score')
	x = G.hub_score()

	print('  A: ' + str(x[a]))
	print('  B: ' + str(x[b]))

	C8A.append(x[a])
	C8B.append(x[b])


	# authority_score
	print(' > authority_score')
	x = G.authority_score()

	print('  A: ' + str(x[a]))
	print('  B: ' + str(x[b]))

	C9A.append(x[a])
	C9B.append(x[b])




	print('\n Anyadimos ' + str(a) + '->' + str(b))
	G.add_edges([(a,b)])

	print('¨' * 60)


feat_mat['N.StrngA'] = C1A
feat_mat['N.StrngB'] = C1B

feat_mat['N.PagerankA'] = C2A
feat_mat['N.PagerankB'] = C2B

feat_mat['N.HarmCenA'] = C3A
feat_mat['N.HarmCenB'] = C3B

feat_mat['N.EigenCenA'] = C4A
feat_mat['N.EigenCenB'] = C4B

feat_mat['N.EccentrA'] = C5A
feat_mat['N.EccentrB'] = C5B

feat_mat['N.CorenessA'] = C6A
feat_mat['N.CorenessB'] = C6B

feat_mat['N.BetweennA'] = C7A
feat_mat['N.BetweennB'] = C7B

feat_mat['N.HubScoreA'] = C8A
feat_mat['N.HubScoreB'] = C8B

feat_mat['N.AuthScoreA'] = C9A
feat_mat['N.AuthScoreB'] = C9B


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~44

feat_mat = feat_mat.drop(['User A'], axis=1)
feat_mat = feat_mat.drop(['User B'], axis=1)

print('\n' + '·'*50 + '\nM E D I D A S   D E   C E N T R A L I D A D\n' + '·'*50)
print(feat_mat)

# Observamos la distribucion de clases
#print(feat_mat['Rating'].value_counts())


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

print('\n' + '·'*50 + '\nM E D I D A S   D E   C E N T R A L I D A D\n' + '·'*50)
print(newX)

print(newX['Rating'].value_counts())


newX.to_csv('dataset.csv', header=True, index=False)

'''
# Clasificadores!!!!

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

X_train, X_test, y_train, y_test = train_test_split(newX, newy, test_size=0.5, random_state=0)


# Naive Bayes 
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d"% (X_test.shape[0], (y_test != y_pred).sum()))
'''