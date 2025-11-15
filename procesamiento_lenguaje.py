"""Importaciones"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

"""Carga de datos"""
#cargar los datos
def read_dataset():
    url_dataset = "./dataset/dataset_humor_train.json"
    dataset = pd.read_json(url_dataset, lines=True)
    #conteo de clases
    print("Total de ejemplos de entrenamiento")
    print(dataset.klass.value_counts())
    # Extracción de los textos en arreglos de numpy
    X = dataset['text'].to_numpy()
    # Extracción de las etiquetas o clases de entrenamiento
    Y = dataset['klass'].to_numpy()
    # Estracción de kis ID (Aún no usado).
    # ID = dataset['id'].to_numpy()
    return X,Y

"""Normalización de datos"""
# TODO: Definir las funciones de preprocesamiento de texto vinculadas al proceso de creación de la matriz 
# Documeno-Término creada con TfidfVectorizer.
def matriz_tfidfVectorizer(X_train, X_val):

    _STOPWORDS = stopwords.words("spanish")  # agregar más palabras a esta lista si es necesario

    # Normalización del texto

    import unicodedata
    import re
    PUNCTUACTION = ";:,.\\-\"'/"
    SYMBOLS = "()[]¿?¡!{}~<>|"
    NUMBERS= "0123456789"
    SKIP_SYMBOLS = set(PUNCTUACTION + SYMBOLS)
    SKIP_SYMBOLS_AND_SPACES = set(PUNCTUACTION + SYMBOLS + '\t\n\r ')

    def normaliza_texto(input_str,
                        punct=False,
                        accents=False,
                        num=False,
                        max_dup=2):
        """
            punct=False (elimina la puntuación, True deja intacta la puntuación)
            accents=False (elimina los acentos, True deja intactos los acentos)
            num= False (elimina los números, True deja intactos los acentos)
            max_dup=2 (número máximo de símbolos duplicados de forma consecutiva, rrrrr => rr)
        """
        
        nfkd_f = unicodedata.normalize('NFKD', input_str)
        n_str = []
        c_prev = ''
        cc_prev = 0
        for c in nfkd_f:
            if not num:
                if c in NUMBERS:
                    continue
            if not punct:
                if c in SKIP_SYMBOLS:
                    continue
            if not accents and unicodedata.combining(c):
                continue
            if c_prev == c:
                cc_prev += 1
                if cc_prev >= max_dup:
                    continue
            else:
                cc_prev = 0
            n_str.append(c)
            c_prev = c
        texto = unicodedata.normalize('NFKD', "".join(n_str))
        texto = re.sub(r'(\s)+', r' ', texto.strip(), flags=re.IGNORECASE)
        return texto


    # Preprocesamiento personalizado 
    def mi_preprocesamiento(texto):
        #convierte a minúsculas el texto antes de normalizar
        tokens = word_tokenize(texto.lower())
        texto = " ".join(tokens)
        texto = normaliza_texto(texto)
        return texto
        
    # Tokenizador personalizado 
    def mi_tokenizador(texto):
        # Elimina stopwords: palabras que no se consideran de contenido y que no agregan valor semántico al texto
        #print("antes: ", texto)
        texto = [t for t in texto.split() if t not in _STOPWORDS]
        #print("después:",texto)
        return texto

    # TODO: Crear la matriz Documento-Término con el dataset de entrenamiento: tfidfVectorizer

    vec_tfidf = TfidfVectorizer(analyzer="word", preprocessor=mi_preprocesamiento, tokenizer=mi_tokenizador,  ngram_range=(1,1))
    X_train_tfidf = vec_tfidf.fit_transform(X_train)

    # Convertir a matriz densa de tipo de dato float32 (tipo de dato por default en Pytorch)
    X_train_tfidf = X_train_tfidf.toarray().astype(np.float32)

    # Tranforma los datos de validación al espacio de representación del entrenamiento
    X_val_tfidf = vec_tfidf.transform(X_val)

    # Convertir a matriz densa de tipo de dato float32 (tipo de dato por default en Pytorch)
    X_val_tfidf = X_val_tfidf.toarray().astype(np.float32)

    return X_train_tfidf, X_val_tfidf, vec_tfidf

"""Traduccion de texto a lenguaje maquina (LabelEncoder)"""
# TODO: Codificar las etiquetas de los datos a una forma categórica numérica: LabelEncoder.
def encode_labels(Y):

    le = LabelEncoder()
    # Normalizar las etiquetas a una codificación ordinal para entrada del clasificador
    Y_encoded = le.fit_transform(Y)
    print("Clases:")
    print(le.classes_)
    print("Clases codificadas:")
    print(le.transform(le.classes_))

    return Y_encoded


"""Generar cunjuntos de datos del dataset"""

def dataset_div(X,Y_encoded):
    # TODO: Dividir el conjunto de datos en conjunto de entrenamiento (80%) y conjunto de pruebas (20%)


    X_train, X_test, Y_train, Y_test =  train_test_split(X, Y_encoded, test_size=0.2, stratify=Y_encoded, random_state=42)

    # Divide el conjunto de entrenamiento en:  entrenamiento (90%) y validación (10%)
    X_train, X_val, Y_train, Y_val =  train_test_split(X_train, Y_train, test_size=0.1, stratify=Y_train, random_state=42)

    # Regresa primero los train y después los test
    data_list_train = [X_train, X_val, Y_train, Y_val]
    data_list_test = [X_test, Y_test]
    
    return data_list_train, data_list_test 

"""Crear mini-batches"""

# Crear minibatches en PyTorch usando DataLoader
def create_minibatches(X, Y, batch_size):
    # Recibe los documentos en X y las etiquetas en Y
    dataset = TensorDataset(X, Y) # Cargar los datos en un dataset de tensores
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # loader = DataLoader(dataset, batch_size=batch_size)
    return loader

"""codificasión de la salida onehot"""

def salida_onehot(Y_train,Y_test,Y_val,NUM_CLASSES = 2):
    
    # Codificación de la salida onehot
    Y_train_one_hot = nn.functional.one_hot(torch.from_numpy(Y_train), num_classes=NUM_CLASSES).float()
    Y_test_one_hot = nn.functional.one_hot(torch.from_numpy(Y_test), num_classes=NUM_CLASSES).float()
    Y_val_one_hot = nn.functional.one_hot(torch.from_numpy(Y_val), num_classes=NUM_CLASSES).float()
    
    return Y_train_one_hot, Y_test_one_hot, Y_val_one_hot  

"""Definicion de la arquitectura"""

# Definir la red neuronal en PyTorch heredando de la clase base de Redes Neuronales: Module
class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        # Definimos la normalizacion de los minibatches
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(p=0.3)
        #self.do = nn.Dropout()

        # Definición de capas, funciones de activación e inicialización de pesos
        input_size_h1 = 256
        input_size_h2 = 128
        input_size_h3 = 64
        input_size_h4 = 32
        

        self.fc1 = nn.Linear(input_size, input_size_h1)
        # PReLU tiene parámetros aprendibles: Se recomienda una función de activación independiente por capa
        self.act1= nn.PReLU()

        self.fc2 = nn.Linear(input_size_h1, input_size_h2)
        # PReLU tiene parámetros aprendibles: Se recomienda una función de activación independiente por capa
        self.act2= nn.PReLU()

        self.fc3 = nn.Linear(input_size_h2, input_size_h3)
        # PReLU tiene parámetros aprendibles: Se recomienda una función de activación independiente por capa
        self.act3= nn.PReLU()

        self.fc4 = nn.Linear(input_size_h3, input_size_h4)
        # PReLU tiene parámetros aprendibles: Se recomienda una función de activación independiente por capa
        self.act4= nn.PReLU()


        # Esta es la salida
        self.output = nn.Linear(input_size_h4, output_size)
        
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.output.weight)

        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)
        if self.fc3.bias is not None:
            nn.init.zeros_(self.fc3.bias)
        if self.fc4.bias is not None:
            nn.init.zeros_(self.fc4.bias)                
        if self.output.bias is not None:
            nn.init.zeros_(self.output.bias)        

    
    def forward(self, X):
        # Definición del orden de conexión de las capas y aplición de las funciones de activación
        x = self.fc1(X)
        x = self.bn1(x) #<-------- Aquí
        x = self.dropout(x)
        x = self.act1(x)
        
        x = self.fc2(x)
        x = self.bn2(x) #<-------- Aquí 
        x = self.dropout(x)
        x = self.act2(x)
        
        x = self.fc3(x)
        x = self.bn3(x) #<-------- Aquí 
        x = self.dropout(x)
        x = self.act3(x)

        x = self.fc4(x)
        x = self.bn4(x) #<-------- Aquí
        x = self.dropout(x) 
        x = self.act4(x)

        x = self.output(x)
        # Nota la última capa de salida 'output' no se activa debido a que CrossEntropyLoss usa LogSoftmax internamente. 
        return x
    
"""Entrenamiento de la red neuronal"""

def train_red_neuronal(X_train_tfidf,X_val_tfidf,output_size=2,epochs=50,learning_rate=0.01,batch_size=128):
    # Establecer los parámetros de la red

    # Parámetros de la red
    input_size =  X_train_tfidf.shape[1]

    #output_size = 2   # 2 clases

    #epochs = 50 # variar el número de épocas, para probar que funciona la programación 
                    # solo usar 2 épocas, para entrenamiento total usar por ejemplo 1000 épocas
    #learning_rate = 0.01 # Generalmente se usan learning rate pequeños (0.001), 

    # Se recomiendan tamaños de batch_size potencias de 2: 16, 32, 64, 128, 256
    # Entre mayor el número más cantidad de memoria se requiere para el procesamiento
    #batch_size = 128 # definir el tamaño del lote de procesamiento 


    # TODO: Convertir los datos de entrenamiento y etiquetas a tensores  de PyTorch

    X_train_t = torch.from_numpy(X_train_tfidf)
    Y_train_t = Y_train_one_hot

    X_val_t = torch.from_numpy(X_val_tfidf)

    # Crear la red
    model = MLP(input_size, output_size)

    # Definir la función de pérdida
    # Mean Square Error (MSE)
    # criterion = nn.MSELoss()
    # criterion = nn.BCELoss()
    weights = torch.tensor([2.5, 1.0]) 
    criterion = nn.CrossEntropyLoss(weight=weights) 

    # Definir el optimizador
    #Parámetros del optimizador: parámetros del modelo y learning rate 
    # Stochastic Gradient Descent (SGD)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Entrenamiento
    print("Iniciando entrenamiento en PyTorch")


    for epoch in range(epochs):
        # Poner el modelo en modo de entrenamiento
        model.train()  
        lossTotal = 0
        #definir el batch_size
        dataloader = create_minibatches(X_train_t, Y_train_t, batch_size=batch_size)

        for X_tr, y_tr in dataloader:
            # inicializar los gradientes en cero para cada época
            optimizer.zero_grad()
            
            # Propagación hacia adelante
            y_pred = model(X_tr)  #invoca al método forward de la clase MLP
            # Calcular el error MSE
            loss = criterion(y_pred, y_tr)
            #Acumular el error 
            lossTotal += loss.item()
            
            # Propagación hacia atrás: cálculo de los gradientes de los pesos y bias
            loss.backward()
            
            # actualización de los pesos: regla de actualización basado en el gradiente:
            #  W = W - learning_rate * dE/dW
            optimizer.step()
            if np.random.random() < 0.1:
                print(f"Batch Error : {loss.item()}")

        print(f"Época {epoch+1}/{epochs}, Pérdida: {lossTotal/len(dataloader)}")

        # Evalúa el modelo con el conjunto de validación
        model.eval()  # Establecer el modo del modelo a "evaluación"
        with torch.no_grad():  # No  calcular gradientes 
            y_pred = model(X_val_t)
            # Aplica softmax para obtener las probabilidades en la evaluación
            y_pred = torch.softmax(y_pred, dim=1)
            # Obtiene una única clase, la más probable
            y_pred = torch.argmax(y_pred, dim=1)        
            print(f"Época {epoch+1}/{epochs}")
            print("P=", precision_score(Y_val, y_pred, average='macro'))
            print("R=", recall_score(Y_val, y_pred, average='macro'))
            print("F1=", f1_score(Y_val, y_pred, average='macro'))
            print("Acc=", accuracy_score(Y_val, y_pred))
    return model

"""Evaluación"""
"""Aquí en está función juntamos la predicción de datos y la evaluación"""

def evaluacion(X_test,vec_tfidf,Y_test,model):
    
    # TODO: Transformar el dataset de test con los mismos preprocesamientos y al  espacio de 
    # representación vectorial que el modelo entrenado, es decir, al espacio de la matriz TFIDF

    # Convertir los datos de prueba a tensores de PyTorch

    X_test_tfidf = vec_tfidf.transform(X_test)

    # Convertir a matriz densa de tipo de dato float32 (tipo de dato por default en Pytorch)
    X_test_tfidf = X_test_tfidf.toarray().astype(np.float32)
    X_t = torch.from_numpy(X_test_tfidf)

    # Desactivar el comportamiento de modo de  entrenamiento: por ejemplo, capas como Dropout
    model.eval()  # Establecer el modo del modelo a "evaluación"

    with torch.no_grad():  # No  calcular gradientes 
        y_pred_test= model(X_t)

    # y_test_pred contiene las predicciones

    # Obtener la clase real
    y_pred_test = torch.argmax(y_pred_test, dim=1)

    print(y_pred_test)

    # TODO: Evaluar el modelo con las predicciones obtenidas y las etiquetas esperadas: 
    # classification_report y  matriz de confusión (métricas Precisión, Recall, F1-measaure, Accuracy)


    print(confusion_matrix(Y_test, y_pred_test))
    print(classification_report(Y_test, y_pred_test, digits=4, zero_division='warn'))




"""Main"""
"""Recopilamos las funciones para crear poder entrenar nuestra red neuronal"""
if __name__ == "__main__":
    print("Empesando el entrenamiento....   ")
    # Leemos los datos del dataset.
    X, Y = read_dataset()

    # Dividimos nuestro dataset.
    Y_encoded = encode_labels(Y=Y)

    data_list_train, data_list_test = dataset_div(X,Y_encoded)

    #Separamos nuestros distintos datos
    X_train = data_list_train[0]
    X_val = data_list_train[1]
    Y_train = data_list_train[2]
    Y_val = data_list_train[3]

    X_test = data_list_test[0]
    Y_test = data_list_test[1]
    

    # Vectorizamos nuestro subconjunto de data set (Y al mismo tiempo lo normalizamos y convertimos en matriz)
    X_train_tfidf, X_val_tfidf, vec_tfidf= matriz_tfidfVectorizer(X_train=X_train,X_val=X_val)


    # Sacamos la codificacion onehot  
    Y_train_one_hot, Y_test_one_hot, Y_val_one_hot  = salida_onehot(Y_train=Y_train, Y_test=Y_test, Y_val=Y_val, NUM_CLASSES = 2) 

    # Entrenamos nuestra red neuronal
    model = train_red_neuronal(X_train_tfidf=X_train_tfidf,X_val_tfidf=X_val_tfidf,output_size=2,epochs=150,learning_rate=0.01,batch_size=128)

    # Imprimimos nuetsra evaluación
    evaluacion(X_test=X_test, vec_tfidf=vec_tfidf, Y_test=Y_test, model=model)

