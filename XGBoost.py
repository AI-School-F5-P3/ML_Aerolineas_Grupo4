import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import  roc_curve, auc, confusion_matrix, precision_score, accuracy_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Obtener la ruta del directorio actual
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construir la ruta al archivo CSV
csv_path = os.path.join(current_dir, 'dataset.csv')

# Cargar data
df = pd.read_csv('dataset.csv')

# Quitar filas  con NaN, columnas id y unnamed
df = df.dropna()
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.loc[:, ~df.columns.str.contains('^id')]

columns = df.drop('satisfaction', axis=1).columns
X = df[columns].copy()  # Crear una copia para evitar SettingWithCopyWarning
y = df['satisfaction']

# Encode categorical variables in X
le_X = LabelEncoder()
for col in columns:
    if X[col].dtype == 'object':
        X[col] = le_X.fit_transform(X[col])

# Encode the target variable
le_y = LabelEncoder()
y = le_y.fit_transform(y)

# Dividir data into training y test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear y entrenar XGBoost model
model = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=10,
    min_child_weight=1,
    subsample=1.0,
    colsample_bytree=1.0,
    random_state=42
)
model.fit(X_train, y_train)

# Hacer predicciones
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
y_test_pred_proba = model.predict_proba(X_test)[:, 1]

# Calcular metricas
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)
train_precision = precision_score(y_train, y_train_pred)
test_precision = precision_score(y_test, y_test_pred)

accuracy = accuracy_score(y_test, y_test_pred)

# Calcular overfitting
overfitting = (train_accuracy - test_accuracy) / train_accuracy * 100

# Calcular ROC 
fpr, tpr, _ = roc_curve(y_test, y_test_pred_proba)
roc_auc = auc(fpr, tpr)

# Calcular la matriz de confusión
cm = confusion_matrix(y_test, y_test_pred)

# Imprimir performance report
print("Model Performance Report")
print("=================================")
print("Precisión del modelo: {:.2f}%".format(accuracy * 100))
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Training Precision: {train_precision:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"AUC-ROC: {roc_auc:.4f}")
print(f"Overfitting: {overfitting:.2f}%")

# Calcular y imprimir feature importance
feature_importance = model.feature_importances_
feature_importance_df = pd.DataFrame({'feature': columns, 'importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance_df)

# Visualizaciones
plt.figure(figsize=(15, 5))

# Feature importance plot
plt.subplot(131)
sns.barplot(x='importance', y='feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')

# ROC curve
plt.subplot(132)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")

# Matriz de confusión
plt.subplot(133)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.show()

print("\nExplicación del Desempeño del Modelo:")
print("1. Precisión: El modelo tiene una precisión de {:.2f}% en el conjunto de prueba, lo que indica su capacidad general para clasificar correctamente la satisfacción del cliente.".format(test_accuracy * 100))
print("2. Exactitud: La exactitud del modelo en el conjunto de prueba es de {:.2f}%, lo que representa la proporción de predicciones positivas correctas.".format(test_precision * 100))
print("3. AUC-ROC: Un valor de {:.2f} sugiere un buen desempeño al discriminar entre clases.".format(roc_auc))
print("4. Sobreajuste: El modelo muestra un sobreajuste del {:.2f}%, que es la diferencia entre el rendimiento de entrenamiento y prueba.".format(overfitting))
print("5. Importancia de las Características: La característica más importante es '{}', lo que sugiere que tiene el mayor impacto en la satisfacción del cliente.".format(feature_importance_df.iloc[0]['feature']))
