# Informe de Rendimiento Detallado para RandomForest

## 1. Métricas de Rendimiento

```
              precision    recall  f1-score   support

           0       0.96      0.98      0.97     11655
           1       0.97      0.94      0.96      9064

    accuracy                           0.96     20719
   macro avg       0.96      0.96      0.96     20719
weighted avg       0.96      0.96      0.96     20719

```

## 2. Matriz de Confusión

![Matriz de Confusión](matriz_confusion_randomforest.png)

Interpretación:
- Verdaderos Positivos: 8527
- Falsos Positivos: 228
- Falsos Negativos: 537
- Verdaderos Negativos: 11427

## 3. Curva ROC

![Curva ROC](curva_roc_randomforest.png)

El área bajo la curva ROC (AUC) es 0.9944, lo que indica un rendimiento del modelo excelente.

## 4. Importancia de Características

![Importancia de Características](importancia_caracteristicas_randomforest.png)

Las 5 características más importantes son:

```
 Importancia    Característica
    0.165157 Característica 11
    0.144552  Característica 6
    0.103235  Característica 4
    0.096559  Característica 3
    0.057262 Característica 13
```

## 5. Resultados de Validación Cruzada

- Puntuación CV Media: 0.9617
- Desviación Estándar: 0.0018

## 6. Análisis de Sobreajuste

- Precisión en Entrenamiento: 1.0000
- Precisión en Prueba: 0.9631
- Diferencia: 0.0369

## Conclusión

El modelo RandomForest muestra un rendimiento general excelente, con un AUC de 0.9944.

No se detecta un sobreajuste significativo.

Las características más importantes para el modelo son ['Característica 11', 'Característica 6', 'Característica 4'], lo que sugiere que estos factores son cruciales para predecir la satisfacción del cliente.

Los resultados de validación cruzada muestran consistencia en el rendimiento del modelo a través de diferentes subconjuntos de datos.
