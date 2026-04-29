import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import requests
from scipy.interpolate import interp1d

# =================================================================
# CONFIGURACIÓN DE DESPLIEGUE (PRODUCCIÓN RENDER)
# =================================================================
# URL del backend en Render (Asegurar que coincida con el nombre del servicio en render.yaml)
API_URL = "https://merschel-raman-api.onrender.com" 

# =================================================================
# CONFIGURACIÓN Y ESTÉTICA (PREMIUM DESIGN)
# =================================================================
st.set_page_config(page_title="Hershell-Raman V1.3", layout="wide", page_icon="🔬")

st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #2e7d32;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# =================================================================
# LÓGICA DE PROCESAMIENTO (RECOLECCIÓN Y ALINEACIÓN LOCAL)
# =================================================================

def load_and_align_spectra(uploaded_files):
    """
    Carga y alinea los espectros localmente antes de enviarlos a la API.
    """
    all_raw_spectra = []
    filenames = []
    
    for uploaded_file in uploaded_files:
        try:
            name = uploaded_file.name
            raw_content = uploaded_file.getvalue().decode('utf-8')
            sep = ',' if (',' in raw_content and raw_content.count(',') > raw_content.count('\t')) else None
            df = pd.read_csv(io.StringIO(raw_content), sep=sep, header=None, engine='python')
            data = df.to_numpy()
            if data.shape[0] < data.shape[1] and data.shape[0] <= 5:
                data = data.T
            spec = data[:, :2].astype(float)
            spec = spec[~np.isnan(spec).any(axis=1)] 
            if len(spec) >= 10:
                all_raw_spectra.append(spec)
                filenames.append(name)
        except: continue

    if not all_raw_spectra: return None, None, None

    mins, maxs = [s[:, 0].min() for s in all_raw_spectra], [s[:, 0].max() for s in all_raw_spectra]
    rango_min_comun, rango_max_comun = max(mins), min(maxs)
    if rango_min_comun >= rango_max_comun: return None, None, None
        
    wavenumbers = np.arange(np.floor(rango_max_comun), np.ceil(rango_min_comun), -1)
    interp_matrix = []
    for spec in all_raw_spectra:
        x, y = spec[:, 0], spec[:, 1]
        if x[0] > x[-1]: x, y = x[::-1], y[::-1]
        f = interp1d(x, y, kind='linear', bounds_error=False, fill_value="extrapolate")
        interp_matrix.append(f(wavenumbers))

    return np.vstack(interp_matrix), wavenumbers, filenames

# =================================================================
# COMUNICACIÓN CON FASTAPI (RENDER)
# =================================================================

def call_fastapi_backend(all_data, wavenumbers, filenames, config):
    """
    Envía los datos a FastAPI en Render con protección contra Cold-Starts.
    """
    spectra_list = []
    for i in range(all_data.shape[0]):
        spectra_list.append({
            "name": filenames[i],
            "x": wavenumbers.tolist(),
            "y": all_data[i, :].tolist()
        })
    
    payload = {
        "spectra": spectra_list,
        "config": config
    }
    
    try:
        # Petición con timeout de 120s para soportar el despertar de Render
        response = requests.post(f"{API_URL}/api/process", json=payload, timeout=120)
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.Timeout:
        st.warning("⏳ El servidor en Render se está encendiendo tras estar en reposo. Esto puede tardar hasta 50 segundos. Por favor, dale clic a 'Procesar en Render' de nuevo.")
        st.stop()
    except requests.exceptions.RequestException as e:
        st.error(f"🚫 Fallo de conexión con el motor matemático. Detalle técnico: {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Fallo crítico inesperado: {str(e)}")
        st.stop()
        
    return None

# =================================================================
# INTERFAZ DE USUARIO (STREAMLIT)
# =================================================================

st.title("🔬 Hershell-Raman V1.3: Distributed (Render)")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Configuración")
    uploaded_files = st.file_uploader("Cargar Espectros (.csv, .txt)", accept_multiple_files=True)
    
    st.subheader("Parámetros del Backend")
    b_algo = st.selectbox("Corrección de Baseline", ["als", "rollingball", "none"])
    s_algo = st.selectbox("Suavizado", ["savgol", "movingavg", "none"])

if not uploaded_files:
    st.info("👋 Bienvenido. Por favor, carga tus archivos espectrales para enviarlos al backend en Render.")
else:
    # Alineación local obligatoria (Hotfix V1.1)
    all_data, wavenumbers, filenames = load_and_align_spectra(uploaded_files)
    
    if all_data is not None:
        st.success(f"✅ {len(filenames)} muestras alineadas localmente. Listas para procesamiento remoto.")
        
        if st.button("🚀 Procesar en Render"):
            config = {"baseline": b_algo, "smoothing": s_algo}
            result = call_fastapi_backend(all_data, wavenumbers, filenames, config)
            
            if result:
                st.success("✅ Procesamiento completado exitosamente en el servidor.")
                # Visualización
                first_spec = result["spectra"][0]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=first_spec["x"], y=first_spec["y"], name=first_spec["name"]))
                fig.update_layout(title=f"Resultado: {first_spec['name']}", xaxis_title="Wavenumber", yaxis_title="Intensidad")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("❌ Error de alineación. Verifica los rangos de tus archivos.")
