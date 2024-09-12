

import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import joblib
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('Dataset/airline_passenger_satisfaction.csv')

columns_to_drop = ['id', 'Unnamed: 0']
df.drop(columns=[col for col in columns_to_drop if col in df.columns], axis=1, inplace=True)

df.dropna(inplace=True) 

X = df.iloc[:, :-1]
y = df['satisfaction']

# Codificación de variables categóricas
le_gender = preprocessing.LabelEncoder()
le_customer_type = preprocessing.LabelEncoder()
le_travel_type = preprocessing.LabelEncoder()
le_class = preprocessing.LabelEncoder()

X['Gender'] = le_gender.fit_transform(X['Gender'])
X['Customer Type'] = le_customer_type.fit_transform(X['Customer Type'])
X['Type of Travel'] = le_travel_type.fit_transform(X['Type of Travel'])
X['Class'] = le_class.fit_transform(X['Class'])


scaler = StandardScaler()
X = scaler.fit_transform(X.astype(float))
le_satisfaction = preprocessing.LabelEncoder()
y = le_satisfaction.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)

# Modelo Sequential de Keras
#  
model = Sequential([
    Input(shape=(X.shape[1],)),  # Input shape
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=1)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {accuracy:.4f}")

# Predicciones
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int).reshape(-1)

# Matriz de Confusión
cm = confusion_matrix(y_test, y_pred_classes)
print("Confusion Matrix:")
print(cm)


class_names = ['neutral or dissatisfied', 'satisfied']

print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes, target_names=class_names))

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# History
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


# Guardamos modelo y scaler
model.save('airline_satisfaction_model.keras')
joblib.dump(scaler, 'nn_scaler.joblib')


print("Model saved successfully.")