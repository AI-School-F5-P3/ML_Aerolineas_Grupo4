# Informe de Optimización de Hiperparámetros con Grid Search para RandomForest

## 1. Introducción

Este informe detalla el rendimiento de un modelo de clasificación basado en **RandomForest** optimizado mediante Grid Search. El objetivo del modelo es predecir la satisfacción del cliente utilizando un conjunto de datos proporcionados. Se analizaron varias combinaciones de hiperparámetros y los resultados se detallan a continuación.

## 2. Mejores Parámetros Encontrados

- **Bootstrap:** False
- **Max Depth:** 30
- **Max Features:** 'sqrt'
- **Min Samples Leaf:** 1
- **Min Samples Split:** 10
- **N Estimators:** 200

Estos parámetros fueron seleccionados como los más óptimos para el modelo de RandomForest, resultando en un buen balance entre complejidad y precisión.

## 3. Rendimiento del Modelo

### 3.1 Precisión Global

- **Precisión (Accuracy):** 96,38%

El modelo alcanza una precisión del 96,38%, lo que indica un rendimiento robusto al predecir la satisfacción del cliente.

### 3.2 Matriz de Confusión

|                | Predicho 0 | Predicho 1 |
|----------------|------------|------------|
| **Real 0**     | 17.225     | 338        |
| **Real 1**     | 787        | 12.729     |

- **Verdaderos Negativos (Real 0, Predicho 0):** 17.225
- **Falsos Positivos (Real 0, Predicho 1):** 338
- **Falsos Negativos (Real 1, Predicho 0):** 787
- **Verdaderos Positivos (Real 1, Predicho 1):** 12.729

### 3.3 Informe de Clasificación

|                         | Precisión | Recall | F1-Score | Soporte |
|-------------------------|-----------|--------|----------|---------|
| **Neutral o Insatisfecho** | 0.96      | 0.98   | 0.97     | 17.563  |
| **Satisfecho**            | 0.97      | 0.94   | 0.96     | 13.516  |
| **Precisión Global**      |           |        | 0.96     | 31.079  |
| **Macro Promedio**        | 0.97      | 0.96   | 0.96     | 31.079  |
| **Promedio Ponderado**    | 0.96      | 0.96   | 0.96     | 31.079  |

## 4. Conclusión

- El modelo RandomForest optimizado mediante Grid Search muestra un excelente rendimiento con una precisión del 96,38%.
- La matriz de confusión muestra un buen equilibrio entre las predicciones correctas y los errores.
- El informe de clasificación indica un rendimiento equilibrado entre ambas clases.

Este modelo puede ser utilizado con confianza en la clasificación de la satisfacción del cliente.
