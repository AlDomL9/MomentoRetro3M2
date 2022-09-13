"""
Regresion Logistica
    Implementacion del algoritmo de regresion logistica con uso de bibliotecas
    de aprendizaje.
    
Autor:
    Alejandro Domi­nguez Lugo
    A01378028
    
Fecha:
    12 de septiembre de 2022
    
"""

#----------------------------------Libreri­as-----------------------------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mlxtend.evaluate import bias_variance_decomp as bias_var # Instalar

#-----------------------------Variables Globales-------------------------------

#------------------------------------Main--------------------------------------
def train (X, Y, num_params = 1, train_size = 0.8, random_state_split = None, 
           shuffle = True, stratify = None, penalty = "l2", dual = False, 
           tol = 0.001, C = 1.0, fit_intercept = True, intercept_scaling = 1, 
           class_weight = None, random_state_regression = None, 
           solver = "lbfgs", max_iter = 100, multi_class = "auto", verbose = 0, 
           warm_start = False, n_jobs = None, l1_ratio = None, 
           loss = "0-1_loss", num_rounds = 200, random_seed = None):
    
    """
    train
        Entrenamiento y evaluación de regresión logística
    
    Argumentos
        X (numpy array):
            Datos de entrada.
        
        Y (numpy array):
            Valores de salida.
        
        num_params (int):
            Número de parámetros del modelo. x >= 1. Default 1.
        
        train_size (int):
            Tamaño de set de entrenamiento. 1 > x > 0. Default 0.8.
        
        random_state_split (int):
            Establece aleatoriedad. Default = None.
            
        shuffle (bool):
            Establece si se realiza o no una mezcla de los datos
        
        stratify (numpy array):
            Los datos se dividen utilizando estos nombres como las clases. 
            Default = None
        
        penalty (string):
            Tipo de penalty. {'none', 'l1', 'l2', 'elasticnet'}. Default = 'l2'
            
        dual (bool):
            Formulación unica o doble. {Solo impementable con 'l2' y 
            'liblinear'}. Default = False
        
        tol (float):
            Tolerancia para el criterio de alto. Default = 0.001
        
        C (float):
            Inverso de la fuerza de la regularización. x > 0. Default = 1
        
        fit_intercept (bool):
            Establece si se identifica una constante. Default = True
        
        intercept_scaling (float):
            Factor de escalamiento del intercepto. si 'liblinear' y 
            fit_interpcept = True. Default = 1
            
        class_weight (diccionario):
            Establece peso de la clase. Formato {clase : peso} o 'balanced'.
            Default = None
        
        random_state_regression (int):
            Establece si se realiza o no una mezcla de los datos. Si 
            'liblinear', 'sag' o 'saga'. Default = None
        
        solver (string):
            Metodo de resolución para modelo. {'newton-cg' - ['l2', 'none'],
            'lbfgs' - ['l2', 'none'], 'liblinear' . ['l1', 'l2'], 'sag' - [
            'l2', 'none'], 'saga' - ['elasticnet', 'l1', 'l2', 'none']}
            
        max_iter (int):
            Cantidad maxima de iteraciones para encontrar modelo. Default = 100
            
        multi_class (string):
            Condiciones para parametros multiples. {'auto', 'ovr', 
            'multinomial'}. Default = 'auto'
        
        verbose (int):
            Para 'liblinear' y 'lbfgs'. x >= 0. Default = 0.
            
        warm_start (bool):
            Reutiliza solución previa para inizialización. Si 'lbfgs', 
            'newton-cg', 'sag' y 'saga'. Default = False
        
        n_jobs (int):
            Numero de cores del CPU a utilizar. -1 = todos. Default = None = 1
        
        l1_ratio (float):
            Cantidad de l1. Si 'elasticnet'. 1 >= x >= 0. Default = None
            
        loss (string):
            Función de perdida para estimacion de bias y varianza. {'0-1_loss',
            'mse'}. Default = '0-1_loss'.
        
        num_roundsds (int):
            Iteraciones para estimación de bias y varianza. x > 0. 
            Default = 200
        
        random_seed (int):
            Estado de aleatoriedad para estimación de bias y varianza. 
            Default = None            
        
    Return:
        model(LogisticRegresion()):
            Modelo generado
            
    """
    # Preparar para escalamiento
    if (num_params == 1):
        X = np.reshape(X, (len(X), 1))
    
    # Escalar
    print("Escalando datos")
    scaler = StandardScaler(copy = False)
    scaler.fit_transform(X)
    print("\tMedia encontrada:    ", scaler.mean_)
    print("\tVarianza encontrada: ", scaler.var_)
    
    # Dividir datos en train y test
    print("\nDividiendo datos en train y test")
    xTrain, xTest, yTrain, yTest = train_test_split(X, Y, 
                                                    train_size = train_size,
                                                    random_state = 
                                                    random_state_split,
                                                    shuffle = shuffle,
                                                    stratify = stratify)
    # Generar modelo
    print("\nPreparando modelo")
    model = LogisticRegression(penalty = penalty, dual = dual, tol = tol, 
                               C = C, fit_intercept = fit_intercept,
                               intercept_scaling = intercept_scaling,
                               class_weight = class_weight, 
                               random_state = random_state_regression,
                               solver = solver, max_iter = max_iter, 
                               multi_class = multi_class, verbose = verbose,
                               warm_start = warm_start, n_jobs = n_jobs,
                               l1_ratio = l1_ratio)
    
    # Calcular bias y varianza del modelo
    print("\nEstimando bias y varianza:")
    mse, bias, var = bias_var(model, xTrain, yTrain, xTest, yTest, loss = loss, 
                              num_rounds = num_rounds, 
                              random_seed = random_seed)
    print("\tBias:     ", bias)
    print("\tVarianza: ", var)
    
    
    # Entrenar modelo
    print("\nEntrenando")
    model.fit(xTrain, yTrain)
    
    # Mostrar características del modelo
    print("\nCaracteristicas finales del modelo:")
    print("\tClases identificadas:     ", model.classes_)
    print("\tCoeficientes:             ", model.coef_)
    print("\tIntercepto:               ", model.intercept_)
    print("\tEvaluacion entrenamiento: ", model.score(xTrain, yTrain))
    print("\tEvaluacion validacion:    ", model.score(xTest, yTest))
    
    # Generar matriz de confucion
    print("\nGenerando Confusion-Matrix")
    cm = confusion_matrix(yTest, model.predict(xTest))

    # Mostrar matriz de confusion
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cm)
    ax.grid(False)
    ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
    ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
    ax.set_ylim(1.5, -0.5)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='white')
    plt.show()
    
    # Mostrar reporte de modelo
    print("\nEvaluacion del modelo: ")
    print(classification_report(yTest, model.predict(xTest)))
    
    
    
    return model

#-----------------------------------Pruebas------------------------------------
#__________________________________Modelo 1____________________________________
# Descargar datos
df= pd.read_csv("./Data/iris.csv")
df.drop(axis = 1, columns = "x0", inplace = True)
df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width",
              "species"]

# Remplazar resultados a valores 1 o 0
df.replace(to_replace = "Iris-setosa", value = 1, inplace = True)
df.replace(to_replace = "Iris-versicolor", value = 0, inplace = True)
df.replace(to_replace = "Iris-virginica", value = 0, inplace = True)

# Entrenar modelo
Y = df["species"].values
X = df["sepal_width"].values

# Generacion y evaluacion del modelo
model1 = train(X, Y, train_size = 0.8, random_state_split = 42, 
              penalty = "none", tol = 0.1, fit_intercept = False,
              random_state_regression = 42, solver = "sag", max_iter = 100, 
              random_seed = 42)

#__________________________________Modelo 2____________________________________
# Mejora de modelo
model2 = train(X, Y, train_size = 0.8, random_state_split = 42, penalty = 'l2',
               tol = 0.001, fit_intercept = True, random_state_regression = 42,
               solver = "saga", max_iter = 100, random_seed = 42)