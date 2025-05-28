# 🧠 Grasp-and-Lift EEG Detection

Este proyecto forma parte de un Trabajo de Fin de Grado (TFG) en Ingeniería Biomédica y se centra en la detección automática de eventos motores relacionados con el agarre y levantamiento de objetos (*grasp-and-lift*) a partir de señales EEG.

Se emplean técnicas de clasificación supervisada (Regresión Logística, Random Forest, XGBoost, MLP), junto con ingeniería de características temporales y frecuenciales. La evaluación del rendimiento se realiza mediante métricas como la curva ROC y el AUC.

---

## 📂 Estructura del repositorio

```
📁 notebooks/
   ├── 01_preprocesamiento.ipynb                       # Limpieza, normalización y ventaneo de etiquetas
   ├── 02_modelado.ipynb                               # Modelado base: LR vs RF
   ├── 03_optimizacion.ipynb                           # Optimización con GridSearch y Optuna
   ├── 04_features_avanzadas.ipynb                     # Ingeniería de características adicionales
   ├── 05_evaluacion_final.ipynb                       # Comparación de modelos base y final
   ├── 06_modelado_avanzado.ipynb                      # Modelado con XGBoost, LGBM, MLP
   ├── 07_seleccion_caracteristicas_avanzada.ipynb     # Selección avanzada de características
   ├── 07b_seleccion_caracteristicas_avanzada_experimentacion.ipynb  # Experimentación con técnicas de selección
   ├── 08_optimizacion_avanzada.ipynb                  # Optimización avanzada con Optuna
   ├── 08b_optimizacion_avanzada_experimentación.ipynb # Variaciones en la búsqueda de hiperparámetros
   ├── 09_postprocesado_autoencoders.ipynb             # Autoencoders para limpieza y reducción

📁 scripts/
   ├── preprocessing.py                                # Funciones reutilizables para preprocesamiento
   ├── feature_engineering.py                          # Funciones de ingeniería de características
   └── models.py                                       # Entrenamiento y evaluación de modelos

📁 test/
   └── test_componentes.py                             # Pruebas unitarias para módulos clave

📁 data/
   └── (no incluida por tamaño; disponible bajo petición)

📁 models/
   └── (no incluida por tamaño; disponible bajo petición)

📄 requirements.txt
📄 requirements_tf.txt
📄 README.md
```

📌 Las carpetas `data/` y `models/` no están incluidas en el repositorio por su tamaño, pero pueden facilitarse si se requieren para la revisión o ejecución completa del proyecto.

📌 Todos los archivos y carpetas ignorados están documentados en el archivo `.gitignore`.

---

## ⚙️ Requisitos

Instalación de dependencias:

```bash
pip install -r requirements.txt
```
```bash
pip install -r requirements_tf.txt
```

---

## 🚀 Cómo ejecutar

1. Clona el repositorio:
   ```bash
   git clone https://github.com/lucyfer28ft/grasp_lift_ai.git
   cd grasp_lift_ai
   ```

2. Abre los notebooks en JupyterLab, VS Code o tu entorno favorito.

3. Ejecuta los cuadernos siguiendo el orden numérico dentro de `notebooks/`.

---

## 📊 Resultados

Los resultados detallados del estudio, incluidas las comparaciones entre modelos y el análisis de eventos motores, se encuentran descritos en la memoria del TFG.

---

## 📚 Referencias

Todas las referencias bibliográficas utilizadas se encuentran debidamente citadas en la memoria del TFG.

---

## 👩‍💻 Autoría

Lucía Fernández Troncoso  
Grado en Ingeniería de la Salud, especialización en Ingeniería Biomédica – Universidad de Sevilla  
TFG 2025

---

## 📝 Licencia

Este repositorio tiene fines exclusivamente académicos. El uso de los datos originales está sujeto a las condiciones de la fuente correspondiente.
