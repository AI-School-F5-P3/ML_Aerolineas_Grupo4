# Informe de Optimización de Hiperparámetros con Grid Search

## 1. Introducción

Este informe detalla el rendimiento de un modelo de clasificación optimizado mediante Grid Search, utilizando un conjunto de datos de satisfacción del cliente. Los mejores parámetros encontrados mediante este proceso se resumen a continuación.

## 2. Mejores Parámetros Encontrados

- **Bootstrap:** False
- **Max Depth:** 30
- **Max Features:** 'sqrt'
- **Min Samples Leaf:** 1
- **Min Samples Split:** 10
- **N Estimators:** 200

Grid Search permitió identificar estos parámetros como los más efectivos para mejorar el rendimiento del modelo.

## 3. Rendimiento del Modelo

### 3.1 Precisión Global

- **Precisión (Accuracy):** 96,38%

El modelo muestra un buen rendimiento global, alcanzando una precisión del 96,38% en el conjunto de prueba.

### 3.2 Matriz de Confusión

|                | Predicho 0 | Predicho 1 |
|----------------|------------|------------|
| **Real 0**     | 17.225     | 338        |
| **Real 1**     | 787        | 12.729     |

- **Verdaderos Negativos (Real 0, Predicho 0):** 17.225
- **Falsos Positivos (Real 0, Predicho 1):** 338
- **Falsos Negativos (Real 1, Predicho 0):** 787
- **Verdaderos Positivos (Real 1, Predicho 1):** 12.729

La matriz de confusión muestra un buen rendimiento en ambas clases, con una ligera tendencia a clasificar erróneamente a clientes satisfechos como insatisfechos.

### 3.3 Informe de Clasificación

|                         | Precisión | Recall | F1-Score | Soporte |
|-------------------------|-----------|--------|----------|---------|
| **Neutral o insatisfecho** | 0.96      | 0.98   | 0.97     | 17.563  |
| **Satisfecho**            | 0.97      | 0.94   | 0.96     | 13.516  |
| **Precisión Global**      |           |        | 0.96     | 31.079  |
| **Macro Promedio**        | 0.97      | 0.96   | 0.96     | 31.079  |
| **Promedio Ponderado**    | 0.96      | 0.96   | 0.96     | 31.079  |

El informe de clasificación muestra que el modelo tiene una alta precisión y recall en ambas clases, con puntuaciones F1 muy cercanas entre sí, indicando un rendimiento equilibrado.

## 4. Conclusión

- El proceso de optimización mediante Grid Search mejoró significativamente la precisión del modelo, alcanzando un 96,38%.
- El modelo muestra un buen rendimiento en la clasificación de ambas clases, con alta precisión y recall.
- La ligera tendencia a clasificar erróneamente a clientes satisfechos como insatisfechos se puede ajustar con más optimización o ajustes adicionales.

Este informe sugiere que el modelo está bien optimizado y es adecuado para su implementación en producción.
