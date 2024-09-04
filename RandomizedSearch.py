from Random_Forest_Classifier import train_random_forest
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Llama a la función para obtener los datos y el modelo entrenado
X_trainset, X_testset, y_trainset, y_testset, rf_model = train_random_forest()

# Definir el rango de hiperparámetros a probar en RandomizedSearchCV
param_dist = {
    'n_estimators': [50, 100, 200, 300],
    'max_features': ['sqrt', 'log2', None],  # Corregido: 'auto' no es válido
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Configurar RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_dist,
    n_iter=100,  # Número de combinaciones a probar
    cv=3,  # Validación cruzada con 3 folds
    verbose=2,
    random_state=3,
    n_jobs=-1  # Usar todos los núcleos del CPU disponibles
)

# Ajustar el modelo utilizando RandomizedSearchCV
random_search.fit(X_trainset, y_trainset)

# Imprimir los mejores hiperparámetros encontrados
print("Mejores parámetros encontrados:", random_search.best_params_)

# Evaluar el modelo con los mejores hiperparámetros
best_rf = random_search.best_estimator_
y_pred = best_rf.predict(X_testset)

# Calcular e imprimir métricas de rendimiento
accuracy = accuracy_score(y_testset, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_testset, y_pred))

print("\nClassification Report:")
print(classification_report(y_testset, y_pred))
