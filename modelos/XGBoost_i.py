import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar los datos
df = pd.read_csv('dataset.csv')

# Eliminar filas con valores NaN y columnas irrelevantes
df = df.dropna()
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.loc[:, ~df.columns.str.contains('^id')]

# Seleccionar columnas específicas y la variable objetivo
columns = df.drop('satisfaction', axis=1).columns
X = df[columns].copy()
y = df['satisfaction']

# Codificar variables categóricas en X
le_X = LabelEncoder()
for col in columns:
    if X[col].dtype == 'object':
        X[col] = le_X.fit_transform(X[col])

# Codificar la variable objetivo
le_y = LabelEncoder()
y = le_y.fit_transform(y)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Aplicar StandardScaler a X_train y X_test
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Guardar el scaler entrenado
joblib.dump(scaler, 'scaler.pkl')

# Crear y entrenar el modelo XGBoost
model = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=10,
    min_child_weight=1,
    subsample=1.0,
    colsample_bytree=1.0,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# Guardar el modelo entrenado con joblib
joblib.dump(model, 'xgboost_model.pkl')

# Hacer predicciones para los conjuntos de entrenamiento y prueba
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)
y_test_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Calcular la precisión
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Imprimir la precisión del modelo
print(f"Precisión del modelo en el conjunto de entrenamiento: {train_accuracy * 100:.2f}%")
print(f"Precisión del modelo en el conjunto de prueba: {test_accuracy * 100:.2f}%")

# Generar el informe de clasificación para el conjunto de prueba
print("Informe de clasificación para el conjunto de prueba:")
print(classification_report(y_test, y_test_pred))

# Matriz de confusión
print("Matriz de confusión para el conjunto de prueba:")
print(confusion_matrix(y_test, y_test_pred))

# Calcular el AUC y la curva ROC
roc_auc = roc_auc_score(y_test, y_test_pred_proba)
print(f"ROC AUC Score: {roc_auc:.4f}")

fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_proba)

# Importancia de características
importance = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': columns,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

print("\nImportancia de las características:")
print(importance_df)

# Visualización de la importancia de características
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Importancia de las Características')
plt.show()

# Guardar los resultados de las predicciones
X_test_df = pd.DataFrame(X_test_scaled, columns=columns)
X_test_df.to_csv('X_test.csv', index=False)
pd.DataFrame(y_test, columns=['satisfaction']).to_csv('y_test.csv', index=False)
