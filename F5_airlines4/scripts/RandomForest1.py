import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Definir paleta de colores basada en el PDF
paleta_colores = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
sns.set_palette(paleta_colores)

# Definir rutas
DIR_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_DATOS = os.path.join(DIR_BASE, 'data')
DIR_MODELOS = os.path.join(DIR_BASE, 'models')
DIR_INFORMES = os.path.join(DIR_BASE, 'informes_modelos')

def asegurar_directorio(directorio):
    if not os.path.exists(directorio):
        os.makedirs(directorio)

def generar_informe_detallado(modelo, X_train, X_test, y_train, y_test, nombre_modelo):
    dir_informe = os.path.join(DIR_INFORMES, nombre_modelo.lower())
    asegurar_directorio(dir_informe)

    # Predicciones
    y_pred = modelo.predict(X_test)
    y_pred_proba = modelo.predict_proba(X_test)[:, 1]

    # Métricas
    precision = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Matriz de Confusión
    mc = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(mc, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusión - {nombre_modelo}')
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')
    plt.savefig(os.path.join(dir_informe, f'matriz_confusion_{nombre_modelo.lower()}.png'))
    plt.close()

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color=paleta_colores[0], lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title(f'Curva ROC - {nombre_modelo}')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(dir_informe, f'curva_roc_{nombre_modelo.lower()}.png'))
    plt.close()

    # Importancia de Características
    importancia_caracteristicas = modelo.feature_importances_
    nombres_caracteristicas = X_test.columns if hasattr(X_test, 'columns') else [f'Característica {i}' for i in range(X_test.shape[1])]
    importancia_caracteristicas_ordenadas = sorted(zip(importancia_caracteristicas, nombres_caracteristicas), reverse=True)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=[imp for imp, _ in importancia_caracteristicas_ordenadas[:10]], 
                y=[name for _, name in importancia_caracteristicas_ordenadas[:10]])
    plt.title(f'Top 10 Características Más Importantes - {nombre_modelo}')
    plt.xlabel('Importancia')
    plt.ylabel('Características')
    plt.savefig(os.path.join(dir_informe, f'importancia_caracteristicas_{nombre_modelo.lower()}.png'))
    plt.close()

    # Validación cruzada
    puntuaciones_cv = cross_val_score(modelo, X_train, y_train, cv=5)
    
    # Generar informe detallado
    informe = f"""# Informe de Rendimiento Detallado para {nombre_modelo}

## 1. Métricas de Rendimiento

```
{classification_report(y_test, y_pred)}
```

## 2. Matriz de Confusión

![Matriz de Confusión](matriz_confusion_{nombre_modelo.lower()}.png)

Interpretación:
- Verdaderos Positivos: {mc[1,1]}
- Falsos Positivos: {mc[0,1]}
- Falsos Negativos: {mc[1,0]}
- Verdaderos Negativos: {mc[0,0]}

## 3. Curva ROC

![Curva ROC](curva_roc_{nombre_modelo.lower()}.png)

El área bajo la curva ROC (AUC) es {roc_auc:.4f}, lo que indica un rendimiento del modelo {'excelente' if roc_auc > 0.9 else 'bueno' if roc_auc > 0.8 else 'aceptable'}.

## 4. Importancia de Características

![Importancia de Características](importancia_caracteristicas_{nombre_modelo.lower()}.png)

Las 5 características más importantes son:

```
{pd.DataFrame(importancia_caracteristicas_ordenadas[:5], columns=['Importancia', 'Característica']).to_string(index=False)}
```

## 5. Resultados de Validación Cruzada

- Puntuación CV Media: {np.mean(puntuaciones_cv):.4f}
- Desviación Estándar: {np.std(puntuaciones_cv):.4f}

## 6. Análisis de Sobreajuste

- Precisión en Entrenamiento: {accuracy_score(y_train, modelo.predict(X_train)):.4f}
- Precisión en Prueba: {precision:.4f}
- Diferencia: {abs(accuracy_score(y_train, modelo.predict(X_train)) - precision):.4f}

## Conclusión

El modelo {nombre_modelo} muestra un rendimiento general {'excelente' if roc_auc > 0.9 else 'bueno' if roc_auc > 0.8 else 'aceptable'}, con un AUC de {roc_auc:.4f}.

{'No se detecta un sobreajuste significativo.' if abs(accuracy_score(y_train, modelo.predict(X_train)) - precision) <= 0.05 else 'Se detecta cierto grado de sobreajuste, considere técnicas de regularización.'}

Las características más importantes para el modelo son {[name for _, name in importancia_caracteristicas_ordenadas[:3]]}, lo que sugiere que estos factores son cruciales para predecir la satisfacción del cliente.

Los resultados de validación cruzada muestran consistencia en el rendimiento del modelo a través de diferentes subconjuntos de datos.
"""

    # Guardar el informe en un archivo Markdown
    with open(os.path.join(dir_informe, f'{nombre_modelo.lower()}_informe_rendimiento.md'), 'w') as f:
        f.write(informe)

    print(f"Informe de rendimiento para {nombre_modelo} generado y guardado en '{dir_informe}'")

    return informe

def entrenar_y_evaluar_modelo():
    # Asegurar que el directorio de informes para RandomForest existe
    dir_informe_randomforest = os.path.join(DIR_INFORMES, 'randomforest')
    asegurar_directorio(dir_informe_randomforest)

    # Cargar dataset
    df = pd.read_csv(os.path.join(DIR_DATOS, 'airline_passenger_satisfaction.csv'))

    # Preprocesamiento de datos
    df = df.drop(['Unnamed: 0', 'id'], axis=1)
    df = df.dropna()
    df['Arrival Delay in Minutes'] = df['Arrival Delay in Minutes'].fillna(df['Arrival Delay in Minutes'].mean())

    # Codificar variables categóricas
    le = LabelEncoder()
    columnas_categoricas = df.select_dtypes(include=['object']).columns
    for col in columnas_categoricas:
        df[col] = le.fit_transform(df[col])

    # Matriz de correlación
    plt.figure(figsize=(20, 16))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Matriz de Correlación')
    plt.savefig(os.path.join(dir_informe_randomforest, 'matriz_correlacion.png'))
    plt.close()

    # Separar características y etiqueta
    X = df.drop('satisfaction', axis=1)
    y = df['satisfaction']

    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Escalado de características
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Entrenar el modelo
    modelo = RandomForestClassifier(n_estimators=1000, random_state=42)
    modelo.fit(X_train, y_train)

    # Generar informe detallado
    generar_informe_detallado(modelo, X_train, X_test, y_train, y_test, "RandomForest")

    # Guardar el modelo entrenado y el scaler
    asegurar_directorio(DIR_MODELOS)
    joblib.dump(scaler, os.path.join(DIR_MODELOS, 'scaler_randomforest1.pkl'))
    joblib.dump(modelo, os.path.join(DIR_MODELOS, 'randomforest_model1.pkl'))

    print("Modelo y scaler guardados exitosamente.")

    return modelo, scaler

def cargar_modelo():
    modelo = joblib.load(os.path.join(DIR_MODELOS, 'randomforest_model1.pkl'))
    scaler = joblib.load(os.path.join(DIR_MODELOS, 'scaler_randomforest1.pkl'))
    return modelo, scaler

if __name__ == "__main__":
    asegurar_directorio(DIR_INFORMES)
    entrenar_y_evaluar_modelo()