import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import  accuracy_score
import xgboost as xgb
import os
import joblib  # Para guardar el scaler y el modelo

# Obtener la ruta del directorio actual
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construir la ruta al archivo CSV
csv_path = os.path.join(current_dir, 'dataset.csv')

# Cargar los datos
df = pd.read_csv('dataset.csv')

# Eliminar filas con valores NaN y columnas irrelevantes
df = df.dropna()
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.loc[:, ~df.columns.str.contains('^id')]


# Seleccionar columnas específicas y la variable objetivo
columns = df.drop('satisfaction', axis=1).columns
X = df[columns].copy()  # Crear una copia para evitar el warning de SettingWithCopy
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

'''
# Guardar el modelo entrenado con joblib
joblib.dump(model, 'xgboost_model.pkl')
'''

# Convertir los arrays escalados de vuelta a DataFrames de pandas antes de guardarlos
X_test_df = pd.DataFrame(X_test_scaled, columns=columns)
X_test_df.to_csv('X_test.csv', index=False)
pd.DataFrame(y_test, columns=['satisfaction']).to_csv('y_test.csv', index=False)

# Crear LabelEncoders
le_customer_type = LabelEncoder()
le_customer_type.fit(df['Customer Type'].dropna().unique())  # Make sure to handle missing values if any

le_travel_type = LabelEncoder()
le_travel_type.fit(df['Type of Travel'].dropna().unique())

le_wifi_service = LabelEncoder()
le_wifi_service.fit(df['Inflight wifi service'].dropna().unique())

le_online_boarding = LabelEncoder()
le_online_boarding.fit(df['Online boarding'].dropna().unique())

le_entertainment = LabelEncoder()
le_entertainment.fit(df['Inflight entertainment'].dropna().unique())
'''
# Guardar LabelEncoders
joblib.dump(le_customer_type, 'le_customer_type.pkl')
joblib.dump(le_travel_type, 'le_travel_type.pkl')
joblib.dump(le_wifi_service, 'le_wifi_service.pkl')
joblib.dump(le_online_boarding, 'le_online_boarding.pkl')
joblib.dump(le_entertainment, 'le_entertainment.pkl')
'''

# Hacer predicciones
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
y_test_pred_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_test_pred)
print(f"Precisión del modelo: {train_accuracy * 100:.2f}%")