# 🔬 Hershell-Raman

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Open Source](https://img.shields.io/badge/Open%20Source-Yes-brightgreen.svg)

**Hershell-Raman** es una plataforma web de código abierto basada en Inteligencia Artificial y el paradigma *Human-in-the-Loop* (HITL) para el análisis quimiométrico, clasificación multivariante y auditoría de espectros Micro-FTIR/Raman.

---

## 🧬 Resumen Científico (Abstract)

El análisis de datos espectrales se enfrenta a un desafío fundamental: la extrema superposición de bandas vibracionales en patógenos cercanamente emparentados, como *Cryptosporidium parvum*, *Giardia lamblia* y *Toxoplasma gondii*. Hershell-Raman aborda y soluciona esta complejidad democratizando el acceso a la quimiometría avanzada de nivel investigativo, sin depender de costosas licencias de software comercial o de infraestructuras cerradas. Mediante la integración de un ensamble de algoritmos de Machine Learning de código abierto, la plataforma no solo aísla y clasifica las huellas espectrales patogénicas, sino que también introduce un novedoso paradigma de "Auditoría de Datos", otorgándole al investigador el control total y transparente sobre las predicciones de la IA.

---

## 🚀 Características Principales (Key Features)

*   **Pipeline Quimiométrico Unificado:** Integración estricta y simétrica para fases de entrenamiento y predicción. Incluye interpolación espectral estandarizada a 1550 variables, Normalización de Variación Normal Estándar (SNV) y la aplicación de la Primera Derivada de Savitzky-Golay (mitigando fluorescencia de fondo y amplificando biomarcadores ocultos).
*   **Ensamble de Machine Learning:** Clasificación fenotípica simultánea a través de modelos matemáticamente ortogonales:
    *   Análisis Discriminante por Mínimos Cuadrados Parciales (PLS-DA).
    *   Máquinas de Soporte Vectorial (MSV) con kernel lineal.
    *   Bosques Aleatorios (Random Forest).
*   **Transparencia Dimensional:** Renderizado inteligente y diferenciado de proyecciones espaciales. La interfaz expone los mapas 2D de variables latentes para PLS-DA y provee una justificación hiperdimensional técnica para algoritmos no proyectivos (MSV y RF) evitando ilusiones ópticas o sesgos espaciales.
*   **Auditoría de Datos (Herramienta Comparador):** Módulo de validación óptica in-situ basado en *Human-in-the-Loop*. Permite la superposición en tiempo real de curvas de la primera derivada para detectar anomalías matemáticas, alertando y confirmando la posible existencia de contaminación cruzada o errores humanos de etiquetado en la fuente.

---

## ⚙️ Arquitectura del Sistema

La arquitectura de software se sostiene sobre una comunicación síncrona Cliente-Servidor (Frontend-Backend) mediada por una **API REST**:

*   **Frontend (Cliente):** Implementado nativamente en Vanilla JS, HTML5 y CSS3, garantizando ligereza y total independencia de frameworks de alto peso. La renderización analítica de los gráficos se delega a `Plotly.js`.
*   **Backend (Servidor):** Construido en Python utilizando `FastAPI`, gestionado mediante el servidor `Uvicorn`. La lógica de validación de los JSON payloads opera bajo el régimen estricto de los modelos estructurados de `Pydantic`.
*   **Motor Matemático:** Todo el poder computacional está vectorizado y optimizado haciendo uso de `scikit-learn`, `SciPy` y `NumPy`.

---

## 💻 Instalación y Despliegue Local

Siga estos pasos para ejecutar el proyecto en su entorno local:

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/jramirezgiraldo-jpg/Merschel-Raman.git
   cd Merschel-Raman
   ```

2. **Crear y activar un entorno virtual (Recomendado):**
   ```bash
   # En Windows
   python -m venv venv
   .\venv\Scripts\activate

   # En Linux/MacOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Asegúrese de que el archivo contenga las dependencias core: `fastapi`, `uvicorn`, `scikit-learn`, `scipy`, `pydantic` y `python-multipart`).*

4. **Ejecutar el servidor Uvicorn:**
   ```bash
   cd backend
   uvicorn main:app --reload
   ```
   *La plataforma estará disponible localmente en `http://localhost:8000/`*

---

## 📋 Uso de la Plataforma

El flujo de trabajo analítico de Hershell-Raman consta de 3 pasos sencillos:

1.  **Cargar Set de Entrenamiento:** Suba sus archivos espectrales en formato CSV (.txt/.csv) clasificados previamente con etiquetas fiables.
2.  **Entrenar Modelo:** Seleccione la arquitectura deseada (PLS-DA, MSV o RF) y presione Entrenar para construir la frontera de decisión en el hiperespacio multivariante.
3.  **Predecir Muestras Ciegas / Validar en Comparador:** Suba espectros de muestras problema o desconocidas para predecir su clasificación. Ante predicciones atípicas, utilice la pestaña **Comparador** para superponer la firma de la muestra contra la huella dactilar de la clase predicha, habilitando la auditoría visual directa.

---

## 📖 Citar este Proyecto (Citation)

Si Hershell-Raman le fue útil para su trabajo de investigación, le agradecemos citar la siguiente publicación:

```bibtex
@article{RamirezGiraldo2026HershellRaman,
  title={Desarrollo y validación de Hershell-Raman: Plataforma web de código abierto basada en IA y Human-in-the-Loop para el análisis quimiométrico de espectros Micro-FTIR},
  author={Ramírez Giraldo, Juan Felipe and Gómez Marín, Jorge Enrique and Arenas, Aylan},
  year={2026},
  journal={TBD}
}
```

---

## 📜 Licencia y Contacto

**Licencia:** Este proyecto se distribuye bajo la [Licencia MIT](https://opensource.org/licenses/MIT). Se permite el uso comercial, modificación, distribución y uso privado, sujeto a la inclusión de los derechos de autor y avisos de licencia.

**Contacto del Desarrollador:**
*   **Juan Felipe Ramírez Giraldo**
*   Grupo de Estudio en Parasitología Molecular (GEPAMOL)
*   GitHub: [@jramirezgiraldo-jpg](https://github.com/jramirezgiraldo-jpg)

> **Agradecimientos:** Este trabajo fue financiado por el Ministerio de Ciencia, Tecnología e Innovación de Colombia (Minciencias) y la Universidad del Quindío.
