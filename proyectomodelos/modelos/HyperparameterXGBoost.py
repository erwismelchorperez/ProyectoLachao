## Hyperparameter optimization using RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix
# import pandas for data wrangling
import pandas as pd
# import numpy for Scientific computations
import numpy as np
# import machine learning libraries
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import RepeatedKFold
import random

from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import balanced_accuracy_score
class HyperparameterXGBoost:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.params = {
            "learning_rate"    : [0.001, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
            "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
            "min_child_weight" : [ i for i in range(0,10)],
            "gamma"            : [i/10.0 for i in range(0,5)],
            "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ],
            "colsample_bylevel": np.arange(0.5, 1.0, 0.1),
            "subsample": [0.5, 1],
            "n_estimators": [100, 250, 500, 750],
            "sampling_method": ["uniform","gradient_based"]
        }
        self.x_train, self.x_validation, self.y_train, self.y_validation = train_test_split(x_train, y_train, test_size = 0.2, random_state=100, stratify=y_train)
        self.x_test = x_test
        self.y_test = y_test
        self.seed = np.random.seed(123)
        """
        self.xgb_params = {
                        "learning_rate": [0.001, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ],
                        "max_depth": [3, 5, 7, 9],
                        "min_child_weight": [3, 5, 7 ,9],
                        "max_delta_step": [3, 5, 7],
                        "sampling_method": ["uniform", "gradient_based"],
                        "n_estimators": [20, 10, 100, 150, 200],
                        "subsample": [0.1, 0.3, 0.5, 0.7, 0.8, 1.0],
        }
        """
        self.xgb_params = {
                        "learning_rate": [0.001, 0.05],
                        "max_depth": [3, 5],
                        "min_child_weight": [3, 5],
                        "max_delta_step": [3, 5],
                        "sampling_method": ["uniform", "gradient_based"],
                        "n_estimators": [150, 200],
                        "subsample": [0.1, 0.3],
        }
        


    def Search_HyperparamterXGBoost(self):
        grid = GridSearchCV(estimator = xgb.XGBClassifier(n_estimators = 10000, random_state = 123),
                            param_grid = self.params,
                            scoring = 'balanced_accuracy',
                            cv = RepeatedKFold(n_splits = 3, n_repeats = 1, random_state = 123),
                            refit = True,
                            verbose = 0,
                            return_train_score = True)
        grid.fit(X = self.x_train, y = self.y_train, **self.params)

        resultados = pd.DataFrame(grid.cv_results)
        resultados.filter(regex= '(param.*|mean_t|std_t)').drop(columns = 'params').sort_values('mean_test_score', ascending = False).head(4)
        resultados.to_csv('data_resultados.csv')

    def Search_HyperparamterXGBoostV2(self):
        xgbr = xgb.XGBClassifier(seed = self.seed, use_label_encoder = False, eval_metric = 'merror')
        xgbr.fit(self.x_train, self.y_train)
        y_predit = xgbr.predict(self.x_test)

        print(xgbr.score(self.x_test, self.y_test))
        xgbc = xgb.XGBClassifier(seed = self.seed, use_label_encoder = False, eval_metric = 'merror')
        grid_search = GridSearchCV(estimator = xgbc, param_grid = self.xgb_params, n_jobs = 1, cv = 3, scoring = 'balanced_accuracy', error_score = 0)
        grid_result = grid_search.fit(self.x_train, self.y_train)

        print("Mejores parametros:    ", grid_result.best_params_)

        final_model = xgbc.set_params(**grid_result.best_params_)
        final_model.fit(self.x_train, self.y_train)
        y_predit = final_model.predict(self.x_test)
        print(xgbc.score(self.x_test, self.y_test))


        confusionmatrix = []
        error_cuadratico = []
        mcc = []
        ber = []
        errorTipoI = []
        errorTipoII = []
        predicciones = []
        for i in range(50):
            final_model = xgb.XGBClassifier(learning_rate = grid_result.best_params_['learning_rate'],
                                            max_depth = grid_result.best_params_['max_depth'],
                                            min_child_weight = grid_result.best_params_['min_child_weight'],
                                            max_delta_step = grid_result.best_params_['max_delta_step'],
                                            sampling_method = grid_result.best_params_['sampling_method'],
                                            n_estimators = grid_result.best_params_['n_estimators'],
                                            subsample = grid_result.best_params_['subsample'])
            final_model.fit(self.x_train, self.y_train)
            predict = final_model.predict(self.x_test)

            predicciones.append(predict)


            matriz = confusion_matrix(self.y_test, predict)
            errortipoI = matriz[0][1]/(matriz[0][0] + matriz[0][1])
            errortipoII = matriz[1][0]/(matriz[1][0] + matriz[1][1])

            confusionmatrix.append(matriz)
            error_cuadratico.append(mean_squared_error(self.y_test, predict ))
            mcc.append(matthews_corrcoef( self.y_test, predict))
            ber.append(balanced_accuracy_score( self.y_test, predict ))
            errorTipoI.append(errortipoI)
            errorTipoII.append(errortipoII)

        return (grid_result.best_params_,confusionmatrix,error_cuadratico, mcc, ber, errorTipoI, errorTipoII, predicciones)
