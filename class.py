import matplotlib.pyplot as plt
import pandas            as pd
import numpy             as np

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network  import MLPClassifier
from sklearn.naive_bayes     import BernoulliNB
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.dummy           import DummyClassifier
from sklearn.tree            import DecisionTreeClassifier

from scipy.stats             import f_oneway
from scipy                   import stats



# Leemos la matriz de caracteristicas
data = pd.read_csv('dataset.csv')

# Separamos la variable clase
X = data.drop(['Rating'], axis=1)
y = data['Rating']


scores = []
measures = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

#print(sorted(sklearn.metrics.SCORERS.keys()))


# CLASIFICADOR NAIVE BAYES
print('\n ' + '-'*50 + '\n · Clasificador Naive Bayes\n')

# Tuning de parametros
# params_nv = {'alpha': [0.01, 0.1, 0.5, 1.0, 10.0]}
# grid_nv = GridSearchCV(BernoulliNB(), params_nv, cv=10, scoring='accuracy')
# grid_nv.fit(X,y)
# print(grid_nv.best_params_)

clf = BernoulliNB(alpha=0.01)
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



# CLASIFICADOR RED NEURONAL
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




'''
# Shapiro Wilk
print('\n\n' + '~'*50 + '\n · S H A P I R O   W I L K\n' + '~'*50)

class_names = ['Naive Bayes', 'kNN', 'Random Forest']
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 score']

for i in range(len(scores)):

	print('\n ' + class_names[i].upper() + '\n')

	for j in range(len(scores[0])):

		print(' · ' + class_names[i] + ' - ' + metric_names[j])

		shapiro_test = stats.shapiro(scores[i][j])

		tstatis = shapiro_test.statistic
		pvalue  = shapiro_test.pvalue

		print('    > Shapiro Test Statistic: ' + str(tstatis))
		print('    > Shapiro Test PValue:    ' + str(pvalue))

		if pvalue > 0.1:
			print('     (Distribucion normal!)\n')
		else:
			print('     (OJO!!!)\n')

# ANOVA
print('\n\n' + '~'*50 + '\n · A N O V A\n' + '~'*50)

for j in range(len(scores[0])):

	print(' · ' + metric_names[j])
	print('\n    Naive Bayes mean: ' + str(np.mean(scores[0][j])))
	print('    kNN mean:         ' + str(np.mean(scores[1][j])) + '\n')

	tstatis, pvalue = f_oneway(scores[0][j], scores[1][j])

	print('    > ANOVA Test Statistic: ' + str(tstatis))
	print('    > ANOVA Test PValue:    ' + str(pvalue))

	if pvalue > 0.1:
		print('     (No hay diferencia...)\n')
	else:
		print('     (Hay diferencia!!!)\n')
'''