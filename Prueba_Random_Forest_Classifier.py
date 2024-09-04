import numpy as np 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
import joblib

def train_random_forest():
    file_id = "1oKFnhKBtO_-eEYenjplsVJAzbcAOYspq"
    url = f"https://drive.google.com/uc?id={file_id}"
    df = pd.read_csv(url)

    columns_to_drop = ['id', 'Unnamed: 0']
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], axis=1, inplace=True)
    df.dropna(inplace=True) 

    X = df.iloc[:, :-1]
    y = df['satisfaction']

    # Label encoders para variables categóricas
    le_gender = preprocessing.LabelEncoder()
    le_customer_type = preprocessing.LabelEncoder()
    le_travel_type = preprocessing.LabelEncoder()
    le_class = preprocessing.LabelEncoder()

    # Transformación de variables categóricas  
    X['Gender'] = le_gender.fit_transform(X['Gender'])
    X['Customer Type'] = le_customer_type.fit_transform(X['Customer Type'])
    X['Type of Travel'] = le_travel_type.fit_transform(X['Type of Travel'])
    X['Class'] = le_class.fit_transform(X['Class'])

    # Normalizamos el dataset de training
    X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

    X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

    # Instancia del Clasificador Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=3)

    # Entrenamiento del Random Forest classifier
    rf.fit(X_trainset, y_trainset)

    # Retornamos el modelo entrenado y los conjuntos de datos
    return X_trainset, X_testset, y_trainset, y_testset, rf
