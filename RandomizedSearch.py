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
best_params = random_search.best_params_
print("Mejores parámetros encontrados:", best_params)

# Evaluar el modelo con los mejores hiperparámetros
best_rf = random_search.best_estimator_
y_pred = best_rf.predict(X_testset)

# Calcular e imprimir métricas de rendimiento
accuracy = accuracy_score(y_testset, y_pred)
conf_matrix = confusion_matrix(y_testset, y_pred)  # Asignamos la matriz de confusión
class_report = classification_report(y_testset, y_pred)  # Asignamos el reporte de clasificación

print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
print(classification_report(y_testset, y_pred))

# Guardar el informe en un archivo de texto
with open('random_search_report.txt', 'w') as f:
    f.write(f"Mejores parámetros encontrados: {best_params}\n\n")
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(conf_matrix))
    f.write("\n\nClassification Report:\n")
    f.write(class_report)

print("Informe guardado en random_search_report.txt")