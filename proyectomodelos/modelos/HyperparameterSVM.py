## Hyperparameter optimization using RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix

# import pandas for data wrangling
import pandas as pd
# import numpy for Scientific computations
import numpy as np
# import machine learning libraries
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import RepeatedKFold

from scipy.stats import randint as sp_randint
import scipy.stats as stats
import random
from sklearn.svm import SVC
from collections import Counter
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import balanced_accuracy_score

class HyperparameterSVM:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.space = {
            'C': stats.uniform(0,50),
            "kernel":['linear','poly','rbf','sigmoid'],
            "gamma": np.arange(0.1, 1, 0.2)
        }
        self.param_grid = {'C': [0.1, 1, 10, 100, 1000], 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['linear','poly','rbf','sigmoid']}
        self.param_gridnew = {'C': [0.01, 1.0],'kernel': ['linear','poly','rbf','sigmoid'],'gamma':['scale']}
        self.Kernel = ['linear','poly','rbf','sigmoid']
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def Search_HyperparamterSVM(self):
        n_iter_search=20
        clf = SVC(gamma='scale')
        #Random = RandomizedSearchCV(clf, param_distributions=self.space,n_iter=n_iter_search,cv=3,scoring='balanced_accuracy')
        Random = GridSearchCV(
                            estimator  = SVC(kernel= "rbf", gamma='scale'),
                            param_grid = self.param_gridnew,
                            scoring    = 'accuracy',
                            n_jobs     = -1,
                            cv         = 5, 
                            verbose    = 0,
                            return_train_score = True)
        Random.fit(self.x_train, self.y_train)
        print(Random.best_params_)
        print("Accuracy:"+ str(Random.best_score_))

        confusionmatrix = []
        error_cuadratico = []
        mcc = []
        ber = []
        errorTipoI = []
        errorTipoII = []
        predicciones = []
        for i in range(50):
            clf = SVC(C= Random.best_params_['C'], kernel = Random.best_params_['kernel'], gamma = Random.best_params_['gamma'])
            clf.fit(self.x_train, self.y_train)
            predict = clf.predict(self.x_test)
            
            predicciones.append(predict)

            matriz = confusion_matrix(self.y_test, predict)
            errortipoI = matriz[0][1]/(matriz[0][0] + matriz[0][1])
            errortipoII = matriz[1][0]/(matriz[1][0] + matriz[1][1])

            confusionmatrix.append(matriz)
            error_cuadratico.append(mean_squared_error(self.y_test, predict))
            mcc.append(matthews_corrcoef(self.y_test, predict))
            ber.append(balanced_accuracy_score(self.y_test, predict))
            errorTipoI.append(errortipoI)
            errorTipoII.append(errortipoII)

        return (Random.best_params_,confusionmatrix,error_cuadratico, mcc, ber, errorTipoI, errorTipoII, predicciones)
