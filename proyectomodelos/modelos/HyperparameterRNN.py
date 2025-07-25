import tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score, StratifiedKFold

from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


import numpy as np
import pandas as pd

from collections import Counter
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import balanced_accuracy_score
class HyperparameterRNN:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.initializer = tf.keras.initializers.GlorotNormal()
        self.metricas = [keras.metrics.TruePositives(name='tp'),
                keras.metrics.FalsePositives(name='fp'),
                keras.metrics.TrueNegatives(name='tn'),
                keras.metrics.FalseNegatives(name='fn'),
                keras.metrics.BinaryAccuracy(name='accuracy'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc'),
                keras.metrics.AUC(name='prc', curve='PR')]
        self.earlystopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy', verbose=1,patience=10,mode='max',restore_best_weights=True)
        self.x_train, self.x_validation, self.y_train, self.y_validation = train_test_split(x_train, y_train, test_size = 0.2, random_state=100, stratify=y_train)
        self.x_test = x_test
        self.y_test = y_test
        self.space = {
            "optimizer":hp.choice('optimizer',['adam', 'rmsprop', 'sgd', 'adamax', 'adagrad']),
            "activation1":hp.choice('activation1',['relu', 'tanh', 'sigmoid', 'softmax' ]),
            "activation2":hp.choice('activation2',['relu', 'tanh', 'sigmoid', 'softmax' ]),
            "activation3":hp.choice('activation3',['relu', 'tanh', 'sigmoid', 'softmax' ]),
            "salida":hp.choice('salida',['relu', 'tanh', 'sigmoid', 'softmax' ]),
            'batch_size': hp.quniform('batch_size', 16, 64, 16),
            'neuronsh1': hp.quniform('neuronsh2', 2, 100, 10),
            'neuronsh2': hp.quniform('neuronsh1', 2, 100, 10),
            'neuronsh3': hp.quniform('neuronsh3', 2, 100, 10),
            'epochs': hp.quniform('epochs', 20, 50, 10),
            'patience': hp.quniform('patience', 3, 20, 3),
        }
        self.optimizadores = ['adam', 'rmsprop', 'sgd', 'adamax', 'adagrad']
        self.activadores = ['relu', 'tanh', 'sigmoid', 'softmax' ]


    def ANN(self, optimizer = 'sgd', neuronsh1=32, neuronsh2= 6, neuronsh3= 6, batch_size=32, epochs=20, activation1='relu',
                activation2='tanh', activation3='sigmoid', salida='softmax', patience=3, loss='categorical_crossentropy'):
        model = Sequential()
        model.add(Dense(neuronsh1, kernel_initializer=self.initializer, input_shape=(self.x_train.shape[1],), activation=activation1))
        model.add(Dense(neuronsh2, activation=activation2))
        model.add(Dense(neuronsh3, activation=activation3))
        model.add(Dense(2,activation=salida))
        model.compile(optimizer = optimizer, loss=loss)
        early_stopping = EarlyStopping(monitor="loss", patience = patience)# early stop patience

        history = model.fit(self.x_train, to_categorical(self.y_train,num_classes=2,dtype ="int32"),
                  batch_size = batch_size,
                  epochs = epochs,
                  callbacks = [early_stopping],
                  validation_data = (self.x_validation, to_categorical(self.y_validation,num_classes=2,dtype ="int32")),
                  verbose=0)

        return model

    def objective(self, params):
        params = {
            "optimizer":str(params['optimizer']),
            "activation1":str(params['activation1']),
            "activation2":str(params['activation2']),
            "activation3":str(params['activation3']),
            "salida":str(params['salida']),
            'batch_size': abs(int(params['batch_size'])),
            'neuronsh1': abs(int(params['neuronsh1'])),
            'neuronsh2': abs(int(params['neuronsh2'])),
            'neuronsh3': abs(int(params['neuronsh3'])),
            'epochs': abs(int(params['epochs'])),
            'patience': abs(int(params['patience']))
        }
        clf = KerasClassifier(build_fn=self.ANN,**params, verbose=0)
        kfold_validacion = KFold(10)

        score = -np.mean(cross_val_score(clf, self.x_train, self.y_train, cv=kfold_validacion, scoring="balanced_accuracy"))
        print('loss',score, 'status', STATUS_OK )
        return {'loss':score, 'status': STATUS_OK }

    def Search_HyperparamterRNN(self):
        mcd = 10
        kfold_validacion = KFold(10)
        clf = KerasClassifier(build_fn=self.ANN, verbose=0)
        scores = cross_val_score(clf, self.x_train, self.y_train, cv=kfold_validacion, scoring= "balanced_accuracy")
        print(scores)
        print("Accuracy:"+ str(scores.mean()) )

        best = fmin(fn=self.objective,
            space=self.space,
            algo=tpe.suggest,
            max_evals=10)
        print("Mejor modelo:   ",best)

        confusionmatrix = []
        error_cuadratico = []
        mcc = []
        ber = []
        errorTipoI = []
        errorTipoII = []
        predicciones = []
        for i in range(50):
            modelo = self.ANN(optimizer = self.optimizadores[best['optimizer']], neuronsh1 = int(best['neuronsh1']), neuronsh2 = int(best['neuronsh2']),
                        neuronsh3 = int(best['neuronsh3']), batch_size = int(best['batch_size']), epochs = int(best['epochs']), activation1 = self.activadores[best['activation1']],
                        activation2 = self.activadores[best['activation2']], activation3 = self.activadores[best['activation3']], salida = self.activadores[best['salida']],
                        patience = best['patience'], loss='categorical_crossentropy')

            predict = modelo.predict(self.x_test)
            predicciones.append(predict.argmax(axis = 1))

            matriz = confusion_matrix(to_categorical(self.y_test,num_classes=2,dtype ="int32").argmax(axis = 1) , predict.argmax(axis = 1))
            errortipoI = matriz[0][1]/(matriz[0][0] + matriz[0][1])
            errortipoII = matriz[1][0]/(matriz[1][0] + matriz[1][1])

            confusionmatrix.append(matriz)
            error_cuadratico.append(mean_squared_error(to_categorical(self.y_test,num_classes=2,dtype ="int32").argmax(axis = 1) , predict.argmax(axis = 1) ) )
            mcc.append(matthews_corrcoef( to_categorical(self.y_test,num_classes=2,dtype ="int32").argmax(axis = 1) , predict.argmax(axis = 1) ))
            ber.append(balanced_accuracy_score( to_categorical(self.y_test,num_classes=2,dtype ="int32").argmax(axis = 1) , predict.argmax(axis = 1) ))
            errorTipoI.append(errortipoI)
            errorTipoII.append(errortipoII)

        return (best,confusionmatrix,error_cuadratico, mcc, ber, errorTipoI, errorTipoII, predicciones)
