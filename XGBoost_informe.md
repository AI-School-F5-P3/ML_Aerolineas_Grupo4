# Informe de Rendimiento del Modelo XGBoost: Clasificador de Satisfacción del Cliente

## 1. Introducción

Este informe detalla el rendimiento de un modelo de clasificación XGBoost aplicado a un conjunto de datos de satisfacción del cliente. Evaluamos métricas clave incluyendo precisión, informe de clasificación, matriz de confusión, puntuación ROC AUC e importancia de características.

## 2. Rendimiento del Modelo

### 2.1 Precisión

- **Precisión en Conjunto de Entrenamiento:** 98,21%
- **Precisión en Conjunto de Prueba:** 96,37%

El modelo muestra un fuerte rendimiento tanto en el conjunto de entrenamiento como en el de prueba, con solo una pequeña caída en la precisión cuando se aplica a datos no vistos.

### 2.2 Informe de Clasificación

|       | Precisión | Recall | Puntuación-F1 | Soporte |
|-------|-----------|--------|---------------|---------|
| **0** | 0,96      | 0,98   | 0,97          | 17583   |
| **1** | 0,97      | 0,94   | 0,96          | 13496   |
| **Exactitud**       |           |        | 0,96          | 31079   |
| **Macro Promedio**   | 0,97      | 0,96   | 0,96          | 31079   |
| **Prom. Ponderado**  | 0,96      | 0,96   | 0,96          | 31079   |

El modelo demuestra un rendimiento equilibrado en ambas clases, con alta precisión, recall y puntuaciones F1 tanto para clientes satisfechos (1) como para neutrales/insatisfechos (0).

### 2.3 Matriz de Confusión

|                | Predicho 0 | Predicho 1 |
|----------------|------------|------------|
| **Real 0**     | 17241      | 342        |
| **Real 1**     | 786        | 12710      |

- **Verdaderos Negativos** (Neutral/insatisfecho correctamente predicho): 17.241
- **Falsos Positivos** (Neutral/insatisfecho predicho como satisfecho): 342
- **Falsos Negativos** (Satisfecho predicho como neutral/insatisfecho): 786
- **Verdaderos Positivos** (Satisfecho correctamente predicho): 12.710

La matriz de confusión muestra que el modelo funciona bien en la clasificación de ambas clases, con una ligera tendencia a clasificar erróneamente a clientes satisfechos como neutrales/insatisfechos.

### 2.4 Puntuación ROC AUC

- **Puntuación ROC AUC:** 0,9951

La alta puntuación AUC indica una excelente capacidad discriminativa entre las dos clases.

## 3. Importancia de Características

Las 10 características más importantes para predecir la satisfacción del cliente son:

1. Embarque en línea (0,419225)
2. Tipo de Viaje (0,179675)
3. Servicio wifi en vuelo (0,122342)
4. Tipo de Cliente (0,051813)
5. Entretenimiento a bordo (0,045859)
6. Clase (0,037584)
7. Servicio de check-in (0,020254)
8. Ubicación de la puerta (0,014759)
9. Limpieza (0,013270)
10. Comodidad del asiento (0,013253)

El embarque en línea, el tipo de viaje y el servicio wifi en vuelo son los tres principales factores que influyen en las predicciones de satisfacción del cliente.

## 4. Conclusiones

- El modelo XGBoost demuestra una alta precisión (96,37% en el conjunto de prueba) en la predicción de la satisfacción del cliente.
- Muestra un rendimiento equilibrado en ambas clases, con alta precisión y recall.
- La capacidad discriminativa del modelo es excelente, como lo demuestra la alta puntuación ROC AUC de 0,9951.
- El embarque en línea, el tipo de viaje y el servicio wifi en vuelo son los factores más cruciales para determinar la satisfacción del cliente.
- Hay una ligera indicación de overfitting, dada la diferencia del 1,84% entre la precisión de entrenamiento y prueba. Sin embargo, esta diferencia no es grave.
