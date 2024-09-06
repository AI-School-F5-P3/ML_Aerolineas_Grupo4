import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Definir las rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

def train_and_evaluate_model():
    # Cargar el dataset
    df = pd.read_csv('/Users/jyajuber/Factoriaf5/Proyecto4/F5_airlines4/data/airline_passenger_satisfaction.csv')

    # Preprocesamiento de datos
    df = df.drop(['Unnamed: 0', 'id'], axis=1)
    df = df.dropna()

    # Codificación de variables categóricas
    le = LabelEncoder()
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df[col] = le.fit_transform(df[col])

    # Separar características y etiqueta
    X = df.drop('satisfaction', axis=1)
    y = df['satisfaction']

    # Dividir en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Escalado de características
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Entrenamiento del modelo final (usando 1000 árboles como en el ejemplo original)
    model = GradientBoostingClassifier(n_estimators=1000, random_state=42)
    model.fit(X_train, y_train)

    # Predicciones
    y_pred = model.predict(X_test)

    # Evaluación del modelo
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print(classification_report(y_test, y_pred))

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Matriz de Confusión')
    plt.savefig('confusion_matrix_gb.png')
    plt.close()

    # Importancia de características
    feature_importance = model.feature_importances_
    feature_importance_sorted = sorted(zip(feature_importance, X.columns), reverse=True)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=[imp for imp, _ in feature_importance_sorted[:10]], 
                y=[name for _, name in feature_importance_sorted[:10]])
    plt.title('Top 10 Características Más Importantes')
    plt.xlabel('Importancia')
    plt.ylabel('Características')
    plt.savefig('feature_importance_gb.png')
    plt.close()

    # Guardar el modelo entrenado y el scaler
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler_GradientBoosting1.pkl'))
    joblib.dump(model, os.path.join(MODELS_DIR, 'GradientBoosting_model1.pkl'))

    print("Modelo y scaler guardados exitosamente.")

    return model, scaler

def load_model():
    model = joblib.load(os.path.join(MODELS_DIR, 'GradientBoosting_model1.pkl'))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler_GradientBoosting1.pkl'))
    return model, scaler

if __name__ == "__main__":
    train_and_evaluate_model()