# SOCIAL NETWORK ANALYSIS - EMRD
# Monica Ramperez Andres


import matplotlib.pyplot as plt
import pandas            as pd
import numpy             as np

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network  import MLPClassifier
from sklearn.naive_bayes     import GaussianNB
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.dummy           import DummyClassifier
from sklearn.tree            import DecisionTreeClassifier

from scipy.stats             import shapiro, f_oneway, kruskal



# Leemos la matriz de caracteristicas
data = pd.read_csv('feature_matrix.csv')

# Separamos la variable clase
X = data.drop(['Rating'], axis=1)
y = data['Rating']


scores = []
measures = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']


# CLASIFICADORES
print('\n\n' + '~'*50 + '\n · C L A S I F I C A D O R E S\n' + '~'*50)


# CLASIFICADOR NAIVE BAYES
print('\n ' + '-'*50 + '\n · Clasificador Naive Bayes\n')

# Tuning de parametros
# params_nv = {'var_smoothing': [0.0, 0.01, 0.1, 0.5, 1.0]}
# grid_nv = GridSearchCV(GaussianNB(), params_nv, cv=10, scoring='accuracy')
# grid_nv.fit(X,y)
# print(grid_nv.best_params_)

clf = GaussianNB(var_smoothing=0.0)
aux_scores = []

for i in measures:
	aux_scores.append(cross_val_score(clf, X, y, scoring=i, cv=10))
	print('    > ' + i + ': ' + str(aux_scores[-1]) + '      (MEAN: ' + str(round(aux_scores[-1].mean(), 2)) + ')')

scores.append(aux_scores)



# CLASIFICADOR kNN
print('\n ' + '-'*50 + '\n · Clasificador kNN\n')

# Tuning de parametros
# params_knn = {'n_neighbors':[1,5,7,10], 'metric':['minkowski', 'jaccard', 'matching']}
# grid_knn = GridSearchCV(KNeighborsClassifier(), params_knn, cv=10, scoring='accuracy')
# grid_knn.fit(X,y)
# print(grid_knn.best_params_)


clf = KNeighborsClassifier(n_neighbors=1, metric='minkowski')
aux_scores = []

for i in measures:
	aux_scores.append(cross_val_score(clf, X, y, scoring=i, cv=10))
	print('    > ' + i + ': ' + str(aux_scores[-1]) + '      (MEAN: ' + str(round(aux_scores[-1].mean(), 2)) + ')')

scores.append(aux_scores)



# CLASIFICADOR DECISION TREE
print('\n ' + '-'*50 + '\n · Decision Tree\n')

# Tuning de parametros
# params__dt = {'criterion':['gini','entropy'],'max_depth':[5,10,30,70,120]}
# grid__dt = GridSearchCV(RandomForestClassifier(), params__dt,cv=10, scoring='accuracy')
# grid__dt.fit(X, y)
# print(grid__dt.best_params_)


clf = DecisionTreeClassifier(criterion='entropy', max_depth=30)

aux_scores = []

for i in measures:
	aux_scores.append(cross_val_score(clf, X, y, scoring=i, cv=10))
	print('    > ' + i + ': ' + str(aux_scores[-1]) + '      (MEAN: ' + str(round(aux_scores[-1].mean(), 2)) + ')')

scores.append(aux_scores)



# CLASIFICADOR RADOM FOREST
print('\n ' + '-'*50 + '\n · Clasificador Random Forest\n')

# Tuning de parametros
# params_rf = {'n_estimators': [550,600,650], 'max_depth': [14,16,18]}
# grid_rf = GridSearchCV(RandomForestClassifier(), params_rf,cv=10, scoring='accuracy')
# grid_rf.fit(X, y)
# print(grid_rf.best_params_)

clf = RandomForestClassifier(max_depth=18, n_estimators=550)
aux_scores = []

for i in measures:
	aux_scores.append(cross_val_score(clf, X, y, scoring=i, cv=10))
	print('    > ' + i + ': ' + str(aux_scores[-1]) + '      (MEAN: ' + str(round(aux_scores[-1].mean(), 2)) + ')')

scores.append(aux_scores)



# CLASIFICADOR RED NEURONAL - MLP
print('\n ' + '-'*50 + '\n · MLPClassifier\n')

# Tuning de parametros
# params_mlp = {'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
# 		      'activation': ['tanh', 'relu'],
# 		      'solver': ['sgd', 'adam'],
# 		      'alpha': [0.0001, 0.05],
# 		      'learning_rate': ['constant','adaptive']}
# grid_mlp = GridSearchCV(MLPClassifier(), params_mlp,cv=10, scoring='accuracy')
# grid_mlp.fit(X, y)
# print(grid_mlp.best_params_)


clf = MLPClassifier(activation='tanh', alpha=0.0001, 
	                hidden_layer_sizes=(50, 100, 50), 
	                learning_rate='adaptive', solver='adam')
aux_scores = []

for i in measures:
	aux_scores.append(cross_val_score(clf, X, y, scoring=i, cv=10))
	print('    > ' + i + ': ' + str(aux_scores[-1]) + '      (MEAN: ' + str(round(aux_scores[-1].mean(), 2)) + ')')

scores.append(aux_scores)



# TEST ESTADISTICOS
print('\n\n' + '~'*50 + '\n · T E S T S   E S T A D I S T I C O S\n' + '~'*50)

class_names = ['Naive Bayes', 'kNN', 'Decision Tree', 'Random Forest', 'MLP']
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 score', 'roc_auc']


for clasA in range(5):

	for clasB in range(clasA + 1, 5):

		print('\n' + '*'*40 + '\n ' + class_names[clasA] + '/' + class_names[clasB] + '\n')

		for j in range(len(scores[0])):

			print(' · ' + class_names[clasA] + '/' + class_names[clasB] + ' - (' + metric_names[j] +')')

			if np.mean(scores[clasA][j]) > np.mean(scores[clasB][j]):
				print('    Parece mejor ' + class_names[clasA])
			else:
				print('    Parece mejor ' + class_names[clasB])


			# Realizamos el test de normalidad
			tstatis1, pvalue1 = shapiro(scores[clasA][j])
			tstatis2, pvalue2 = shapiro(scores[clasB][j])

			#print('    > Shapiro Test Statistic A: ' + str(tstatis))
			#print('    > Shapiro Test PValue A:    ' + str(pvalue))

			#print('    > Shapiro Test Statistic B: ' + str(tstatis))
			#print('    > Shapiro Test PValue B:    ' + str(pvalue))


			# Si se asume normalidad se utiliza ANOVA
			if pvalue1 > 0.05 and pvalue2 > 0.05:

				tstatis, pvalue = f_oneway(scores[clasA][j], scores[clasB][j])

				#print('    > ANOVA Test Statistic: ' + str(tstatis))
				#print('    > ANOVA Test PValue:    ' + str(pvalue))

				if pvalue > 0.05:
					print('     ANOVA: No hay diferencias significativas...')
				else:
					print('     ANOVA: Hay diferencias significativas!!!')



			# Si no se asume normalidad se utiliza Krusal Wallis
			else:

				tstatis, pvalue = kruskal(scores[clasA][j], scores[clasB][j])

				#print('    > KRUSAL W. Test Statistic: ' + str(tstatis))
				#print('    > KRUSAL W. Test PValue:    ' + str(pvalue))

				if pvalue > 0.05:
					print('     KRUSAL W.: No hay diferencias significativas...')
				else:
					print('     KRUSAL W.: Hay diferencias significativas!!!')

			
			print('~'*50)