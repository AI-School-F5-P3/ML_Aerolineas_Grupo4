

### Informe del Rendimiento del Modelo: Clasificador de Red Neuronal

---

## 1. Introducción
Este informe detalla el rendimiento de un modelo de clasificación basado en una Red Neuronal aplicada a un conjunto de datos de satisfacción del cliente. Se incluyen métricas clave como la precisión, matriz de confusión, informe de clasificación, así como un análisis de sobreajuste y la arquitectura del modelo.

## 2. Métricas Generales del Modelo

- **Precisión del Clasificador en Test**: 0.9597 (95.97%)
- **Precisión del Clasificador en Entrenamiento**: 0.9714 (97.14%)
- **Diferencia de Precisión (Overfitting estimado)**: 0.0117 (1.20%)

Estas métricas muestran que el modelo tiene un rendimiento sólido en términos de precisión tanto en los datos de entrenamiento como de prueba, con una diferencia mínima que sugiere un ligero sobreajuste.

## 3. Matriz de Confusión

La matriz de confusión resume el rendimiento del modelo de la siguiente manera:


|   | Predicción: Satisfecho | Predicción: Neutral o Insatisfecho |
| --- | --- | --- |
| **Real: Satisfecho** | 17,153 | 410 |
| **Real: Neutral o Insatisfecho** | 841 | 12,675 |



Esto indica que el modelo clasifica correctamente la mayoría de los ejemplos de ambas clases, con una pequeña cantidad de falsos negativos y falsos positivos.

## 4. Informe de Clasificación

| Clase                     | Precisión | Recall | F1-Score | Soporte |
|---------------------------|-----------|--------|----------|---------|
| Satisfecho                 | 0.95      | 0.98   | 0.96     | 17,563  |
| Neutral o Insatisfecho     | 0.97      | 0.94   | 0.95     | 13,516  |
| **Exactitud Total**        |           |        | **0.96** | 31,079  |
| Promedio Macro             | 0.96      | 0.96   | 0.96     | 31,079  |
| Promedio Ponderado         | 0.96      | 0.96   | 0.96     | 31,079  |

## 5. Arquitectura del Modelo

El modelo está compuesto por una red neuronal secuencial con la siguiente arquitectura:

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Capa (Tipo)                     ┃ Salida                 ┃ Parámetros    ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ densa (Dense)                   │ (None, 64)             │         1,472 │
│ densa_1 (Dense)                 │ (None, 32)             │         2,080 │
│ densa_2 (Dense)                 │ (None, 16)             │           528 │
│ densa_3 (Dense)                 │ (None, 1)              │            17 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
Total parámetros: 12,293
```

## 6. Conclusiones

El modelo de Red Neuronal muestra un rendimiento excelente en términos de precisión y generalización. La pequeña diferencia entre las métricas de entrenamiento y prueba indica un mínimo sobreajuste. La estructura de la red es relativamente simple, lo cual ha sido suficiente para capturar la complejidad del problema.
