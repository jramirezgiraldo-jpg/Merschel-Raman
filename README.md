# Merschel-Raman V8.2

Plataforma profesional para el análisis quimiométrico avanzado de espectros Raman y FTIR, especializada en la detección y clasificación de parásitos (Toxoplasma, Cryptosporidium, Giardia, etc.) mediante modelaje multivariado y machine learning de estado sólido.

## Características

- **Pre-procesamiento dinámico:** Líneas base Whittaker-Eilers (ALS), Rolling Ball, filtros Savitzky-Golay, primera/segunda derivada, normalizaciones SNV, Min-Max, área L1.
- **Microanálisis de picos:** Algoritmos de prominencia y detección automática basada en bases de datos funcionales.
- **Exploración Quimiométrica No Supervisada:** PCA (Principal Component Analysis), HCA (Dendrogramas) y matrices de correlación de Pearson.
- **Modelos de Clasificación Supervisados:** PLS-DA, Redes SVM y Random Forest implementados desde el backend en Scikit-Learn.
- **Carga en masa y predicción ciega:** Construcción de modelo incremental a partir de lotes CSV/TXT bi-direccionales y predicción masiva de muestras.

## Estructura del Repositorio

- `/public`: Archivos del front-end en JavaScript, HTML, CSS (Plotly.js, Math.js).
- `/backend`: Lógica pesada, motor estadístico en Python alimentado por FastAPI.

## Instalación y Uso (Desarrollo Local)

1. **Clonar repositorio e instalar dependencias del backend:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```
2. **Ejecutar servidor FastAPI:**
   ```bash
   uvicorn main:app --reload
   ```
3. El frontend está diseñado para consumir la API montada en el puerto `8000`.

## Despliegue (Github Pages + Cloud Backend)
El sistema está configurado para ajustar dinámicamente sus terminales:
- Si opera en `.github.io`, la aplicación envía sus tensores por POST al servidor en la nube definido en `API_BASE_URL`.
- En el ambiente local, las solicitudes se redirigen a `localhost:8000`.

---
*Desarrollado para fines de investigación académica en Biofotónica y Espectroscopía.*
