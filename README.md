# ml-llm-challenge
ML challenge and LLM-RAG

# Machine Learning y LLM con RAG

Este repositorio contiene el desarrollo de dos retos orientados a Machine Learning y Procesamiento de Lenguaje Natural, organizados en las siguientes carpetas:

1. **db-ml-classification**
   - **Objetivo:** Resolver un reto de clasificación utilizando algoritmos de Machine Learning con el dataset [Wine](https://archive.ics.uci.edu/dataset/109/wine) disponible en UCI Machine Learning Repository.
   - **Contenido:**
     - `db-ml-classification/Pre Analisis Wine Classification.ipynb`: Notebook que incluye el analisis exploratorio del dataset y la prueba de varios algoritmos.
     - `db-ml-classification/wine_classification.ipynb`: Notebook que incluye el desarrollo del proceso de clasificación.
     - `db-ml-classification/Wine Classification KNN - Final.ipynb`: Notebook que incluye el desarrollo del reto con el algoritmo k-nearest neighbors (KNN) 
     - `db-ml-classification/Wine Classification Naive Bayes  - Final.ipynb`: Notebook que incluye el desarrollo del reto con el algoritmo Naive Bayes
     - `db-ml-classification/Wine Classification Neural Net - Final.ipynb`: Notebook que  incluye el desarrollo del reto con el algoritmo  Neural Net (Neural network models - MLPClassifier) 
     - `db-ml-classification/Wine Classification SVM  - Final.ipynb`: Notebook que  incluye el desarrollo del reto con el algoritmo Support Vector Machines 
     - `db-ml-classification/Predicciones.ipynb`: Notebook que incluye el uso de los modelos anteriormente entrenados usando MLFlow.
     - `db-ml-classification/PrTabular_Generator_Tensorflow.ipynb`: Notebook que incluye el proceso de implementación de un GAN para generar datos sinteticos para el dataset de Wine
     - `db-ml-classification/Tabular_Generator_SDV_.ipynb`: Notebook que incluye el proceso de implementación de la libreria [SDV](https://sdv.dev/) para generar datos sinteticos para el dataset de Wine
     - `db-ml-classification/Wine Clasiffication - Docs.pdf`: Documentación detallada sobre el enfoque utilizado, resultados obtenidos y conclusiones.
   - **Descripción del Reto:** Este reto se enfoca en el análisis y clasificación de tipos de vino a partir de características químicas, utilizando técnicas de Machine Learning para optimizar la precisión del modelo.

2. **resume-screen**
   - **Objetivo:** Implementar un proceso de selección de currículums utilizando un modelo de Lenguaje de gran tamaño (LLM).
   - **Contenido:**
     - `Resume_Screening_LLM.ipynb`: Notebook que muestra el desarrollo del proceso de selección de currículums.
     - `Resume Screening LLM .pdf`: Documentación detallada que explica el flujo de trabajo, los modelos utilizados y los resultados.
   - **Descripción del Reto:** En este reto se busca automatizar el proceso de selección de currículums aplicando técnicas de Procesamiento de Lenguaje Natural (NLP) y Large Language Model, empleando modelos avanzados de lenguaje para filtrar y seleccionar candidatos potenciales.

## Estructura del Proyecto
```lua
.
├── db-ml-classification/
│   ├── Pre Analisis Wine Classification.ipynb`:
│   ├── wine_classification.ipynb`:
│   ├── Wine Classification KNN - Final.ipynb`:
│   ├── Wine Classification Naive Bayes  - Final.ipynb`:
│   ├── Wine Classification Neural Net - Final.ipynb`:
│   ├── Wine Classification SVM  - Final.ipynb`:
│   ├── Predicciones.ipynb`:
│   ├── PrTabular_Generator_Tensorflow.ipynb`:
│   ├── Tabular_Generator_SDV_.ipynb`:
│   ├── Wine Clasiffication - Docs.pdf`:
│   ├── wine_classification.ipynb
│   ├── documentation.pdf
│
└── resume-screen/
│    ├── resume-screen/Resume_Screening_LLM.ipynb    
│    ├── resume-screen/Resume Screening LLM .pdf 
```

## Requisitos

- Python 3.8+
- Librerías requeridas: [Ver dentro de los notebooks]
- Entorno recomendado: [Databricks para MLFlow]

## Ejecución

1. Clonar este repositorio:  
   ```bash
   git clone <URL del repositorio>
   ```

2. Navegar a la carpeta del reto deseado e iniciar Jupyter Notebook o Databricks para ejecutar los notebooks:
    ```bash
    cd db-ml-classification
    jupyter notebook Pre Analisis Wine Classification.ipynb
   ```
    ```bash
    cd resume-screen
    jupyter notebook Resume_Screening_LLM.ipynb
   ```

## Contribuciones
Las contribuciones son bienvenidas. Por favor, sigue el formato de commits y abre un Pull Request para revisión.