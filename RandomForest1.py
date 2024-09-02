from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Cargar el archivo desde Google Drive
file_id = "1oKFnhKBtO_-eEYenjplsVJAzbcAOYspq"
url = f"https://drive.google.com/uc?id={file_id}"
df = pd.read_csv(url)

# Preprocesamiento de datos
df = df.drop(columns=['Unnamed: 0', 'id'])
df = df.dropna()  # Eliminar filas con valores faltantes
df['Arrival Delay in Minutes'] = df['Arrival Delay in Minutes'].fillna(df['Arrival Delay in Minutes'].mean())

# Codificación de variables categóricas
le = LabelEncoder()
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# Separar características (X) y la variable objetivo (y)
X = df.drop('satisfaction', axis=1)
y = df['satisfaction']

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalado de características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Lista de diferentes números de árboles para probar
n_estimators_list = [10, 50, 100, 200, 500, 1000]

# Inicializar listas para almacenar los resultados
accuracy_list = []
roc_auc_list = []
confusion_matrices = []
feature_importances_list = []
cv_scores_list = []

# Bucle para entrenar y evaluar el modelo con diferentes números de árboles
for n_estimators in n_estimators_list:
    print(f"\nEntrenando RandomForest con {n_estimators} árboles...")
    
    # Entrenamiento del modelo
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluación con Validación Cruzada
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    cv_scores_list.append(cv_scores)
    
    # Predicciones
    y_pred = model.predict(X_test)
    
    # Evaluación del modelo
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
    accuracy_list.append(accuracy)
    roc_auc_list.append(roc_auc)
    
    # Almacenar la matriz de confusión
    confusion_matrices.append(confusion_matrix(y_test, y_pred))

    # Almacenar la importancia de características
    feature_importances_list.append(model.feature_importances_)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print(classification_report(y_test, y_pred))
    
    # Evaluación de overfitting
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")
    print(f"Diferencia (indicación de overfitting): {train_accuracy - test_accuracy:.4f}")

# Convertir la lista de importancias a un DataFrame
feature_importances_df = pd.DataFrame(feature_importances_list, columns=X.columns)

# Verificar que no hay NaN en el DataFrame antes de calcular la media
print(feature_importances_df.isna().sum())  # Esto debe mostrar ceros si todo está bien

# Calcular la importancia media de cada característica
mean_importances = feature_importances_df.mean().sort_values(ascending=False)

print("\nImportancia Media de Características:")
print(mean_importances)

# Gráfico de Accuracy vs n_estimators
plt.figure(figsize=(12, 6))
plt.plot(n_estimators_list, accuracy_list, marker='o', linestyle='-', color='b', label='Accuracy')
plt.title('Accuracy vs Número de Árboles (n_estimators)')
plt.xlabel('Número de Árboles (n_estimators)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.show()

# Gráfico de ROC AUC vs n_estimators
plt.figure(figsize=(12, 6))
plt.plot(n_estimators_list, roc_auc_list, marker='o', linestyle='-', color='g', label='ROC AUC Score')
plt.title('ROC AUC vs Número de Árboles (n_estimators)')
plt.xlabel('Número de Árboles (n_estimators)')
plt.ylabel('ROC AUC Score')
plt.grid(True)
plt.legend()
plt.show()

# Graficar la importancia media de las características
plt.figure(figsize=(12, 8))
sns.barplot(x=mean_importances.values, y=mean_importances.index)
plt.title('Importancia Media de las Características')
plt.xlabel('Importancia')
plt.ylabel('Características')
plt.show()

# Visualización de la Matriz de Confusión para el último modelo entrenado
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrices[-1], annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Matriz de Confusión (n_estimators={n_estimators_list[-1]})')
plt.show()
