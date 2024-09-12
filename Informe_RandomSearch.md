# Informe de Optimización de Hiperparámetros con Random Search: Modelo Random Forest

## 1. Introducción

Este informe detalla el rendimiento de un modelo **Random Forest** optimizado mediante **Random Search** para la predicción de la satisfacción del cliente. A través de Random Search, se encontraron los hiperparámetros más adecuados para mejorar la precisión del modelo.

## 2. Mejores Parámetros Encontrados

- **N Estimators:** 300
- **Min Samples Split:** 5
- **Min Samples Leaf:** 1
- **Max Features:** 'sqrt'
- **Max Depth:** 30
- **Bootstrap:** False

Random Search optimizó estos hiperparámetros para maximizar la precisión del modelo.

## 3. Rendimiento del Modelo

### 3.1 Precisión Global

- **Precisión (Accuracy):** 96,35%

El modelo alcanza una precisión del 96,35%, lo que muestra un buen rendimiento en la predicción de la satisfacción del cliente.

### 3.2 Matriz de Confusión

|                | Predicho 0 | Predicho 1 |
|----------------|------------|------------|
| **Real 0**     | 17.216     | 347        |
| **Real 1**     | 788        | 12.728     |

- **Verdaderos Negativos (Real 0, Predicho 0):** 17.216
- **Falsos Positivos (Real 0, Predicho 1):** 347
- **Falsos Negativos (Real 1, Predicho 0):** 788
- **Verdaderos Positivos (Real 1, Predicho 1):** 12.728

El modelo clasifica correctamente la mayoría de los casos, con ligeros errores en la predicción de clientes satisfechos.

### 3.3 Informe de Clasificación

|                         | Precisión | Recall | F1-Score | Soporte |
|-------------------------|-----------|--------|----------|---------|
| **Neutral o insatisfecho** | 0.96      | 0.98   | 0.97     | 17.563  |
| **Satisfecho**            | 0.97      | 0.94   | 0.96     | 13.516  |
| **Precisión Global**      |           |        | 0.96     | 31.079  |
| **Macro Promedio**        | 0.96      | 0.96   | 0.96     | 31.079  |
| **Promedio Ponderado**    | 0.96      | 0.96   | 0.96     | 31.079  |

El modelo muestra un rendimiento equilibrado en ambas clases.

## 4. Conclusión

- El modelo de **Random Forest** optimizado mediante Random Search ofrece una precisión del 96,35%, lo que indica un buen rendimiento general.
- Aunque la precisión es levemente menor que la obtenida con Grid Search, sigue siendo altamente competitiva.
- La matriz de confusión muestra una ligera tendencia a errores en la clasificación de clientes satisfechos, pero el rendimiento global es sólido.

El modelo optimizado es adecuado para su uso en producción para predecir la satisfacción del cliente.
