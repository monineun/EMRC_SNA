import matplotlib.pyplot as plt
import pandas            as pd
import numpy             as np

from sklearn.model_selection import cross_val_score
from sklearn.neural_network  import MLPClassifier
from sklearn.naive_bayes     import GaussianNB
from sklearn.ensemble        import RandomForestClassifier
from sklearn.dummy           import DummyClassifier
from sklearn                 import neighbors

from scipy.stats             import f_oneway
from scipy                   import stats


# Clasificadores!!!!

data = pd.read_csv('dataset.csv')

X = data.drop(['Rating'], axis=1)
y = data['Rating']


scores = []

# Naive Bayes 
print('\n ' + '-'*50 + '\n · Clasificador Naive Bayes\n')

clf = GaussianNB()
aux_scores = []

aux_scores.append(cross_val_score(clf, X, y, scoring='accuracy', cv=10))
print('    > Accuracy: ' + str(aux_scores[-1]))

aux_scores.append(cross_val_score(clf, X, y, scoring='precision', cv=10))
print('    > Precision: ' + str(aux_scores[-1]))

aux_scores.append(cross_val_score(clf, X, y, scoring='recall', cv=10))
print('    > Recall: ' + str(aux_scores[-1]))

aux_scores.append(cross_val_score(clf, X, y, scoring='f1', cv=10))
print('    > F1 score: ' + str(aux_scores[-1]))

scores.append(aux_scores)

# kNN
print('\n ' + '-'*50 + '\n · Clasificador kNN\n')

clf = neighbors.KNeighborsClassifier(3)
aux_scores = []

aux_scores.append(cross_val_score(clf, X, y, scoring='accuracy', cv=10))
print('    > Accuracy: ' + str(aux_scores[-1]))

aux_scores.append(cross_val_score(clf, X, y, scoring='precision', cv=10))
print('    > Precision: ' + str(aux_scores[-1]))

aux_scores.append(cross_val_score(clf, X, y, scoring='recall', cv=10))
print('    > Recall: ' + str(aux_scores[-1]))

aux_scores.append(cross_val_score(clf, X, y, scoring='f1', cv=10))
print('    > F1 score: ' + str(aux_scores[-1]))

scores.append(aux_scores)

'''
# Random Forest
print('\n ' + '-'*50 + '\n · Clasificador Random Forest\n')

clf = RandomForestClassifier()
aux_scores = []

aux_scores.append(cross_val_score(clf, X, y, scoring='accuracy', cv=10))
print('    > Accuracy: ' + str(aux_scores[-1]))

aux_scores.append(cross_val_score(clf, X, y, scoring='precision', cv=10))
print('    > Precision: ' + str(aux_scores[-1]))

aux_scores.append(cross_val_score(clf, X, y, scoring='recall', cv=10))
print('    > Recall: ' + str(aux_scores[-1]))

aux_scores.append(cross_val_score(clf, X, y, scoring='f1', cv=10))
print('    > F1 score: ' + str(aux_scores[-1]))

scores.append(aux_scores)



# Algoritmo basado en reglas?
print('\n ' + '-'*50 + '\n · DummyClassifier\n')

clf = DummyClassifier()


scores = cross_val_score(clf, X, y, scoring='accuracy', cv=10)
print('    > Accuracy: ' + str(scores))

scores = cross_val_score(clf, X, y, scoring='precision', cv=10)
print('    > Precision: ' + str(scores))

scores = cross_val_score(clf, X, y, scoring='recall', cv=10)
print('    > Recall: ' + str(scores))

scores = cross_val_score(clf, X, y, scoring='f1', cv=10)
print('    > F1 score: ' + str(scores))



# Red neuronal
print('\n ' + '-'*50 + '\n · MLPClassifier\n')

clf = MLPClassifier()


scores = cross_val_score(clf, X, y, scoring='accuracy', cv=10)
print('    > Accuracy: ' + str(scores))

scores = cross_val_score(clf, X, y, scoring='precision', cv=10)
print('    > Precision: ' + str(scores))

scores = cross_val_score(clf, X, y, scoring='recall', cv=10)
print('    > Recall: ' + str(scores))

scores = cross_val_score(clf, X, y, scoring='f1', cv=10)
print('    > F1 score: ' + str(scores))


#print(sorted(sklearn.metrics.SCORERS.keys()))
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
