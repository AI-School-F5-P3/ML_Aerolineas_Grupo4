# ✈️ Grupo 4: Proyecto de Aprendizaje Supervisado: Clasificación de Satisfacción del Cliente ✨

## 📝 Descripción del Proyecto

**Objetivo:**  
El proyecto tiene como fin desarrollar un modelo de **aprendizaje supervisado** para predecir la satisfacción de los clientes de **F5 Airlines**, basándose en diversas características. Además, busca identificar las variables más relevantes para la satisfacción del cliente y presentarlas en un informe para el equipo de negocio. El proyecto se divide en dos fases:

1. **Entrenar un modelo predictivo:** Determinar si un cliente estará satisfecho o no con los servicios ofrecidos.
2. **Desarrollar una aplicación interactiva** que permita ingresar datos de nuevos clientes y genere una predicción sobre su nivel de satisfacción.

## 🔧 Tecnologías Utilizadas

- **Scikit-learn** para el entrenamiento y evaluación del modelo.
- **Pandas** para la manipulación y limpieza de datos.
- **Streamlit** para la creación de la aplicación interactiva.
- **Tensorflow y Keras** para entrenar y utilizar la Red Neuronal.
- **Git y GitHub** para el control de versiones.
- **Docker** para contenerizar la aplicación.
- **Azure** para despliegue de la aplicación y la base de datos.
- **MySQL** para el desarrollo de la base de datos.
- **Trello** para la gestión del proyecto.

## 🚀 Niveles de Entrega

### ✔️ Nivel Esencial:
- Modelo funcional de **Machine Learning** que predice el grado de satisfacción de los clientes.
- **Análisis Exploratorio de Datos** (EDA).
- **Overfitting menor al 5%**.
- Productivización del modelo mediante una **aplicación interactiva**.
- **Informe de rendimiento** con métricas como matrices de confusión, curva ROC y análisis de importancia de características.

### ⚙️ Nivel Medio:
- Técnicas de **Ensemble** para mejorar el rendimiento del modelo.
- Uso de **Validación Cruzada** y **Optimización de Hiperparámetros** con técnicas como **Grid Search** y **Random Search**.
- Sistema de **recogida de feedback** y datos para futuros entrenamientos.

### 🖥️ Nivel Avanzado:
- **Contenerización con Docker** para garantizar la portabilidad del modelo.
- **Despliegue en la nube** y almacenamiento de datos recogidos por la aplicación en bases de datos.
- **Test unitarios** (pendientes de implementación).

### 💡 Nivel Experto:
- **Experimentos o despliegues con modelos de redes neuronales** Se ha entrenado un modelo de rede neuronal.
- **Sistemas de MLOps** pendientes de desarrollo para:
   - Entrenamiento y despliegue automático de nuevas versiones del modelo.
   - **A/B Testing** y monitoreo de **Data Drifting** para asegurar la calidad del modelo antes de reemplazarlo.

## 🔍 Resultados Actuales

Hasta el momento, hemos alcanzado todos los niveles esperados, **excepto los test unitarios** y la implementación de un **sistema de MLOps** completo para la actualización automática del modelo en producción.

## 📅 Plazos

**Fecha de entrega:** Jueves 12 de Septiembre. Se presupuestaron dos semanas para la realización de un prototipo funcional.

## 📊 Datos Utilizados

El dataset utilizado contiene información sobre la **satisfacción de los clientes de aerolíneas**, con variables categóricas y numéricas relacionadas con el servicio brindado, tales como el servicio de wifi en vuelo, tipo de viaje, limpieza y comodidad del asiento, entre otros.

### 📂 Descargar Dataset
Puedes descargar el dataset utilizado en el proyecto desde el siguiente enlace:

[Dataset CSV](./modelos/Dataset/)

## 📂 Estructura del Repositorio

- **Modelos de ML**: Entrenamiento de modelos y optimización de hiperparámetros.
- **App Interactiva**: Aplicación para la predicción de satisfacción del cliente.
- **Documentación**: Informes y análisis de resultados.
- **Docker**: Configuración para contenerización de la aplicación.
- **Presentaciones**: Informes técnicos y de negocio para los stakeholders.

## 👥 Colaboración

Este proyecto ha sido desarrollado en equipo utilizando un **flujo de trabajo colaborativo** en GitHub, con ramas específicas para cada funcionalidad y commits bien documentados para asegurar la trazabilidad.

📢 *Contribuidores: F5 AI School - Grupo 4*
