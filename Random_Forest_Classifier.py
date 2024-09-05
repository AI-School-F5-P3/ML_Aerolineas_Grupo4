import numpy as np 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
import matplotlib.pyplot as plt
import joblib


file_id = "1oKFnhKBtO_-eEYenjplsVJAzbcAOYspq"
url = f"https://drive.google.com/uc?id={file_id}"
df = pd.read_csv(url)

# print(df.head())
#df = pd.read_csv('Dataset/airline_passenger_satisfaction.csv')

columns_to_drop = ['id', 'Unnamed: 0']
df.drop(columns=[col for col in columns_to_drop if col in df.columns], axis=1, inplace=True)

#Eliminamos columnas espúreas e innecesariaas
df.dropna(inplace=True) 
X = df.iloc[:, :-1]

y = df['satisfaction']


# label encoders para variables categóricas
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

pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Paso de escalado
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=3))  # Paso de clasificación
])

X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

# Instancia del Clasificador Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=3)

# Entrenamiento del  Random Forest classifier
rf.fit(X_trainset, y_trainset)

# Predición del dataset de testing
predRF = rf.predict(X_testset)

# Entrenar el pipeline en los datos de entrenamiento
pipeline.fit(X_trainset, y_trainset)

# Predecir en el conjunto de prueba
predRF = pipeline.predict(X_testset)

# Crear y ajustar el scaler
scaler = StandardScaler()
scaler.fit(X_trainset)

# Accuracy
accuracy = metrics.accuracy_score(y_testset, predRF)
print("Accuracy del Clasificador Random Forest:", accuracy)

# Confusion Matrix
cm = confusion_matrix(y_testset, predRF)
print("Confusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_testset, predRF))

# Cross-validation score (buscamos si hay overfitting)
cv_scores = cross_val_score(rf, X, y, cv=5)
print("\nCross-validation scores:", cv_scores)
print("CV score Medio:", np.mean(cv_scores))
print("Desviación Estándar de los CV scores:", np.std(cv_scores))

# Calculate training and testing accuracy to assess overfitting
train_accuracy = rf.score(X_trainset, y_trainset)
test_accuracy = rf.score(X_testset, y_testset)
print("\nTraining Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)
print("Diferencia (indicación de overfitting):", train_accuracy - test_accuracy)



feature_names = [ 'Gender', 'Customer Type', 'Age',
                 'Type of Travel', 'Class', 'Flight Distance', 'Inflight wifi service',
                 'Departure/Arrival time convenient', 'Ease of Online booking',
                 'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
                 'Inflight entertainment', 'On-board service', 'Leg room service',
                 'Baggage handling', 'Checkin service', 'Inflight service',
                 'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']

# Importancia de las Características
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Gráfica de Importancia de las Características
plt.figure(figsize=(12, 8))
plt.title("Importancia de las Características")
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()

# Calculamos probabilidades para la curva ROC
y_pred_proba = rf.predict_proba(X_testset)[:, 1]

# Cálculo de la curva ROC 
fpr, tpr, thresholds = metrics.roc_curve(y_testset, y_pred_proba, pos_label='satisfied')

# Calcular AUC
roc_auc = metrics.auc(fpr, tpr)

# Curva ROC 
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='Curva ROC (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa False Positive')
plt.ylabel('Tasa True Positive')
plt.title('Curva Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# Muestra AUC score
print("AUC Score:", roc_auc)

# Guardamos el modelo
joblib.dump(rf, 'rf_model.joblib')
joblib.dump(scaler, 'scaler.pkl')