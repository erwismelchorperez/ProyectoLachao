# Logistic Regression with GridSearchCV
#import nnetsauce as ns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import mean_squared_error

class HyperparameterLR:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test  = x_test
        self.y_test  = y_test
        self.params = {
            'penalty':['l1','l2'],
            'C': np.logspace(-3,3.7),
            'solver':['newton-cg','lbfgs','liblinear']
        }

    def Lineal_Regresion(self):
        lr = LogisticRegression()
        clf = GridSearchCV(lr, param_grid = self.params, scoring='balanced_accuracy', cv = 10)
        clf.fit(self.x_train, self.y_train)

        print("hyperparametros :    ", clf.best_params_)
        print("Accuracy:            ",clf.best_score_)

        confusionmatrix = []
        error_cuadratico = []
        mcc = []
        ber = []
        errorTipoI = []
        errorTipoII = []
        predicciones = []
        for i in range(50):
            lr =LogisticRegression(C=clf.best_params_['C'],penalty=clf.best_params_['penalty'],solver=clf.best_params_['solver'])
            lr.fit(self.x_train, self.y_train)
            predict = lr.predict(self.x_test)
            predicciones.append(predict)
            #print("predict:     ", predict)

            matriz = confusion_matrix(self.y_test, predict)
            errortipoI = matriz[0][1]/(matriz[0][0] + matriz[0][1])
            errortipoII = matriz[1][0]/(matriz[1][0] + matriz[1][1])

            confusionmatrix.append(matriz)
            error_cuadratico.append(mean_squared_error(self.y_test, predict) )
            mcc.append(matthews_corrcoef( self.y_test, predict ))
            ber.append(balanced_accuracy_score( self.y_test, predict ))
            errorTipoI.append(errortipoI)
            errorTipoII.append(errortipoII)

        return (clf.best_params_,confusionmatrix,error_cuadratico, mcc, ber, errorTipoI, errorTipoII, predicciones)

            









