# Banco de Prompts y Metodología Human-in-the-Loop (Material Suplementario)

## Introducción

El presente documento registra de forma detallada las iteraciones y directrices de ingeniería de prompts (*Zero-shot* y *Few-shot*) empleadas durante el desarrollo de la plataforma web **Hershell-Raman**. Esta bitácora sirve como evidencia de la aplicación de la metodología *Human-in-the-Loop* (HITL) para la codificación del backend quimiométrico, la orquestación del frontend interactivo y la resolución de incidencias en entornos de producción. El registro garantiza la transparencia metodológica y facilita la reproducibilidad del ciclo de vida del desarrollo asistido por Inteligencia Artificial.

---

## Categorización del Banco de Prompts

A continuación, se presentan los prompts arquitectónicos más críticos utilizados durante el ciclo de desarrollo del proyecto. Para simplificar la lectura, se ha abstraído la estructura de las instrucciones más extensas conservando el núcleo técnico (Prompts Maestros).

### Fase 1: Arquitectura y Pipeline Quimiométrico

Esta fase aseguró que el procesamiento matemático fuera consistente, simétrico e incluyera rutinas de amplificación de señal antes del entrenamiento.

```markdown
# ORDEN DE DESARROLLO: PIPELINE QUIMIOMÉTRICO UNIFICADO
Actúa como Ingeniero de Datos Quimiométricos. Necesito que crees una función maestra en `main.py` llamada `aplicar_quimiometria(espectros_json)`.

**Instrucciones Obligatorias:**
1. El procesamiento debe ser secuencial y estrictamente simétrico tanto para los datos de entrenamiento como para las predicciones ciegas.
2. Interpola todos los espectros recibidos a exactamente 1550 variables para homogeneizar los ejes X.
3. Aplica Normalización de Variación Normal Estándar (SNV) a las absorbancias interpoladas.
4. OBLIGATORIO: Integra la función `savgol_filter` de SciPy sobre la matriz resultante de SNV. Aplica la Primera Derivada de Savitzky-Golay utilizando una ventana de 15 puntos (`window_length=15`), un polinomio de grado 2 (`polyorder=2`) y calculando la primera derivada (`deriv=1`).
5. Asegúrate de que los endpoints devuelvan la matriz procesada y no la cruda.
```

### Fase 2: Modelos de Clasificación

En esta fase se integraron algoritmos complementarios al modelo base, prestando especial atención a no generar distorsiones visuales en la inferencia dimensional.

```markdown
# ORDEN DE ACTUALIZACIÓN: ENSAMBLE DE MACHINE LEARNING Y TRANSPARENCIA VISUAL
Extiende el motor de clasificación en `main.py`. Actualmente solo tenemos PLS-DA; añade Máquinas de Soporte Vectorial (SVM con kernel lineal) y Bosques Aleatorios (Random Forest) de `scikit-learn`.

**Regla de Negocio para la UI (`main.js`):**
Es crítico evitar la representación de espacios matemáticos artificiales.
- SI el modelo activo es PLS-DA: Permite el renderizado de la gráfica de "Scores" proyectando las primeras Variables Latentes.
- SI el modelo activo es MSV o RF: Oculta el contenedor del gráfico de Plotly y muestra en su lugar un contenedor de texto (`infoContainer`). Inyecta un mensaje que aclare que el modelo está procesando la matriz en el espacio hiperdimensional original de 1550 variables sin reducción de varianza visual.
```

### Fase 3: Auditoría de Datos y Herramienta Óptica

Para dotar al investigador humano de capacidades de auditoría sobre las clasificaciones algorítmicas, se desarrolló un módulo de comparación visual in-situ.

```markdown
# REQUERIMIENTO CIENTÍFICO: HERRAMIENTA DE VALIDACIÓN ÓPTICA (COMPARADOR)
Hemos detectado que algunas muestras ciegas se clasifican inconsistentemente frente a la etiqueta del laboratorio de origen. Necesito integrar un nuevo módulo interactivo en el frontend para realizar auditorías visuales de los datos procesados.

**Instrucción:**
1. Crea una pestaña "Comparador" en `index.html`.
2. Añade dos menús desplegables: uno para seleccionar el "Espectro Promedio" de una clase de entrenamiento y otro para elegir una muestra específica de la tabla de "Predicción de Ciegos".
3. Utiliza Plotly.js para superponer ambas curvas (Absorbancia Procesada vs Número de Onda) en el mismo plano cartesiano.
4. Asegúrate de que el backend provea los datos derivados para confirmar ópticamente si la morfología espectral (huella dactilar) de la muestra anómala coincide con el fenotipo real.
```

### Fase 4: Depuración y Estabilización de Producción

Abordaje directo y estructurado frente a caídas del servidor y discrepancias en los payloads HTTP durante la interacción de la API.

```markdown
# DIAGNÓSTICO DE INCIDENCIA: ERROR HTTP 400 EN COMPARADOR Y ALINEACIÓN DE PAYLOADS
El endpoint `/api/preprocessed_data` está arrojando un error HTTP 400 (Bad Request) al intentar usar el Comparador.

**Causa Raíz:**
El JSON generado por la función `runVisualComparison` en JavaScript no coincide con la validación estricta del modelo Pydantic en FastAPI. El array está enviando objetos con campos faltantes cuando la matriz de datos espectrales está vacía al no resolverse la promesa de lectura.

**Acción Requerida:**
1. Modifica la función en `main.js`. Utiliza `Promise.all()` para iterar sobre los ítems seleccionados.
2. Si las variables `x` e `y` están vacías, implementa un bloque `try-catch` que ejecute `await item.file.text()` para parsear el CSV dinámicamente antes de enviar el payload.
3. Asegura que el diccionario retornado cumpla exactamente con los campos `name`, `x` e `y` definidos en el Pydantic model del backend.
```

---

## Conclusión Metodológica

El paradigma *Human-in-the-Loop* implementado en la construcción de **Hershell-Raman** posicionó a la Inteligencia Artificial estrictamente como un copiloto de codificación. Las decisiones clave relativas a la manipulación matemática de los datos (e.g., parámetros del filtro de Savitzky-Golay, simetría del pipeline y control de proyecciones espaciales) fueron invariablemente establecidas por el criterio analítico humano. La resolución del hallazgo de las muestras anómalas (*Tox5*) no fue delegada al modelo predictivo como una verdad absoluta; por el contrario, la creación guiada de la herramienta de comparación óptica permitió al investigador validar biológica y matemáticamente los falsos positivos, corroborando así que la IA incrementa la eficiencia técnica sin suplantar el raciocinio científico.
