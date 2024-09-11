# Informe de Optimización de Hiperparámetros con Random Search

## 1. Introducción

Este informe detalla el rendimiento de un modelo de clasificación optimizado mediante Random Search, utilizando un conjunto de datos de satisfacción del cliente. Los mejores parámetros encontrados a través de este proceso se detallan a continuación.

## 2. Mejores Parámetros Encontrados

- **N Estimators:** 300
- **Min Samples Split:** 5
- **Min Samples Leaf:** 1
- **Max Features:** 'sqrt'
- **Max Depth:** 30
- **Bootstrap:** False

Random Search identificó estos parámetros como los más efectivos para mejorar el rendimiento del modelo.

## 3. Rendimiento del Modelo

### 3.1 Precisión Global

- **Precisión (Accuracy):** 96,35%

El modelo alcanza una precisión del 96,35%, lo que indica un buen rendimiento en la predicción de la satisfacción del cliente.

### 3.2 Matriz de Confusión

|                | Predicho 0 | Predicho 1 |
|----------------|------------|------------|
| **Real 0**     | 17.216     | 347        |
| **Real 1**     | 788        | 12.728     |

- **Verdaderos Negativos (Real 0, Predicho 0):** 17.216
- **Falsos Positivos (Real 0, Predicho 1):** 347
- **Falsos Negativos (Real 1, Predicho 0):** 788
- **Verdaderos Positivos (Real 1, Predicho 1):** 12.728

La matriz de confusión muestra un rendimiento similar al del modelo optimizado con Grid Search, con una leve tendencia a clasificar erróneamente a clientes satisfechos como neutrales/insatisfechos.

### 3.3 Informe de Clasificación

|                         | Precisión | Recall | F1-Score | Soporte |
|-------------------------|-----------|--------|----------|---------|
| **Neutral o insatisfecho** | 0.96      | 0.98   | 0.97     | 17.563  |
| **Satisfecho**            | 0.97      | 0.94   | 0.96     | 13.516  |
| **Precisión Global**      |           |        | 0.96     | 31.079  |
| **Macro Promedio**        | 0.96      | 0.96   | 0.96     | 31.079  |
| **Promedio Ponderado**    | 0.96      | 0.96   | 0.96     | 31.079  |

El modelo presenta un rendimiento equilibrado en ambas clases, con alta precisión, recall y puntuación F1.

## 4. Conclusión

- El modelo optimizado mediante Random Search alcanza una precisión del 96,35%, lo que demuestra un excelente rendimiento general.
- La matriz de confusión muestra una correcta clasificación en la mayoría de los casos, con una ligera tendencia a errores en la predicción de clientes satisfechos.
- El informe de clasificación confirma que el modelo tiene un rendimiento consistente y equilibrado en ambas clases.

Este informe sugiere que el modelo está bien optimizado y puede ser utilizado en producción para predecir la satisfacción del cliente.
