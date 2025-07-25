## Hyperparameter optimization using RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix
# import pandas for data wrangling
import pandas as pd
# import numpy for Scientific computations
import numpy as np
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import RepeatedKFold

from scipy.stats import randint as sp_randint
import scipy.stats as stats
import random
from collections import Counter
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import balanced_accuracy_score

#Random Forest
from random import randrange as sp_randrange
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from dtreeplt import dtreeplt
from sklearn import tree

class HyperparameterDT:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.rf_params = {
            "max_features": sp_randint(1,x_train.shape[1]),
            'max_depth': sp_randint(5,50),
            "min_samples_split":sp_randint(2,11),
            "min_samples_leaf":sp_randint(1,11),
            'splitter': ['best','random'],
            "criterion":['gini','entropy','log_loss']
        }
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def Search_HyperparamterDT(self):
        n_iter_search=5 #number of iterations is set to 20, you can increase this number if time permits
        clf_dt = tree.DecisionTreeClassifier(random_state=0)
        #arbol = RandomizedSearchCV(clf_dt, param_distributions = self.rf_params, n_iter=n_iter_search, cv=5, scoring='balanced_accuracy')
        arbol = RandomizedSearchCV(clf_dt, param_distributions = self.rf_params, n_iter=n_iter_search, cv=5, scoring='accuracy')

        
        arbol.fit(self.x_train, self.y_train)
        print(arbol.best_params_)
        #print(arbol.cv_results_)
        print("Accuracy:"+ str(arbol.best_score_))

        #print(arbol.best_params_)
        confusionmatrix = []
        error_cuadratico = []
        mcc = []
        ber = []
        errorTipoI = []
        errorTipoII = []
        predicciones = []
        for i in range(50):
            dt = tree.DecisionTreeClassifier(criterion = arbol.best_params_['criterion'],
                                            max_depth = arbol.best_params_['max_depth'],
                                            max_features = arbol.best_params_['max_features'],
                                            min_samples_leaf = arbol.best_params_['min_samples_leaf'],
                                            min_samples_split = arbol.best_params_['min_samples_split'],
                                            splitter = arbol.best_params_['splitter'])

            dt.fit(self.x_train, self.y_train)

            predict = dt.predict(self.x_test)
            predicciones.append(predict)

            matriz = confusion_matrix(self.y_test, predict)
            errortipoI = matriz[0][1]/(matriz[0][0] + matriz[0][1])
            errortipoII = matriz[1][0]/(matriz[1][0] + matriz[1][1])
            #print(matriz[0],"           ",errortipoI,"    ",matriz[1],"           ", errortipoII)

            confusionmatrix.append(matriz)
            error_cuadratico.append(mean_squared_error(self.y_test, predict))
            mcc.append(matthews_corrcoef(self.y_test, predict))
            ber.append(balanced_accuracy_score(self.y_test, predict))
            errorTipoI.append(errortipoI)
            errorTipoII.append(errortipoII)

        return (arbol.best_params_,confusionmatrix,error_cuadratico, mcc, ber, errorTipoI, errorTipoII, predicciones)

