from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from scipy import signal
from pybaselines import Baseline
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelBinarizer
from fastapi.middleware.cors import CORSMiddleware
import re
import pandas as pd
import traceback
from scipy.interpolate import interp1d

def parse_spectroscopy_file(decoded_content: str):
    """
    Extractor Universal: Ignora metadatos, encabezados sucios y delimitadores inconsistentes,
    extrayendo únicamente los valores numéricos de los espectros.
    """
    lines = decoded_content.splitlines()
    cleaned_data = []
    
    for line in lines:
        # Dividir por comas, tabulaciones o múltiples espacios
        parts = re.split(r'[,\t;]+|\s{2,}', line.strip())
        # Filtrar strings vacíos generados por comas finales
        parts = [p.strip() for p in parts if p.strip()]
        
        # Si hay al menos dos valores, intentar convertirlos a flotantes
        if len(parts) >= 2:
            try:
                x = float(parts[0])
                y = float(parts[1])
                cleaned_data.append([x, y])
            except ValueError:
                # Si no son números (ej. 'Time', 'Wavenumber'), se ignora la línea
                continue
                
    if not cleaned_data:
        raise ValueError("No se encontraron datos numéricos válidos en el archivo.")
        
    df = pd.DataFrame(cleaned_data, columns=['Wavenumber', 'Absorbance'])
    # Ordenamiento Monotónico para evitar errores en interpolación
    df = df.sort_values(by='Wavenumber', ascending=True).reset_index(drop=True)
    return df

app = FastAPI(title="Hershell-Raman V8.2 API")

# Configuración de CORS para despliegue en Render
origins = [
    "https://jramirezgiraldo-jpg.github.io",
    "http://localhost:8501", # Streamlit local
    "http://localhost:8000", # FastAPI local / Docs
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos de datos para el Body de la petición (JSON)
class SpectrumData(BaseModel):
    name: str
    x: List[Optional[float]]
    y: List[Optional[float]]

class SpectrumInput(BaseModel):
    name: str
    x: list[Optional[float]]
    y: list[Optional[float]]

class LabeledSpectrumInput(SpectrumInput):
    label: str

class PlsdaRequest(BaseModel):
    spectra: list[LabeledSpectrumInput]
    n_components: int = 2

class ProcessConfig(BaseModel):
    baseline: str = "none"
    smoothing: str = "none"
    # Agregaremos derivadas y normalizaciones después

class ProcessRequest(BaseModel):
    spectra: List[SpectrumData]
    config: ProcessConfig

class ChemoParams(BaseModel):
    range: List[float] = [0.0, 4000.0]
    scale: str = "none"

class ChemoRequest(BaseModel):
    spectra: List[SpectrumInput]
    analysis_type: str = "pca"
    linkage_method: str = "ward"
    color_threshold: Optional[float] = None
    params: ChemoParams = ChemoParams()

class CompareRequest(BaseModel):
    spectra: List[SpectrumData]

# Ruta para servir nuestra UI Front-end
@app.get("/", response_class=HTMLResponse)
async def read_index():
    return FileResponse("../public/index.html")

@app.post("/api/process")
async def process_spectra(request: ProcessRequest):
    try:
        baseline_algo = request.config.baseline
        smoothing_algo = request.config.smoothing
        
        # 1. Alinear todos los espectros al rango común (Evitar length mismatch)
        # build_symmetric_matrix ya implementa el Hotfix de ordenamiento y rango común
        Y_matrix, x_ref = build_symmetric_matrix(request.spectra)
        
        # 2. Creación de la matriz final NUEVA (Master DataFrame)
        # Filas = Muestras, Columnas = Wavenumbers
        df_final = pd.DataFrame(Y_matrix, columns=x_ref)
        
        baseline_fitter = Baseline()
        processed_results = []
        
        # 3. Paso al pre-procesamiento iterativo sobre la matriz alineada
        for idx, row in df_final.iterrows():
            y = row.values
            x = x_ref # Eje X Maestro
            
            # Corrección de Línea Base
            if baseline_algo == "als":
                y_base, _ = baseline_fitter.asls(y, lam=1e5, p=0.01)
                y = y - y_base
            elif baseline_algo == "rollingball":
                half_window = max(5, len(y) // 20)
                y_base, _ = baseline_fitter.rolling_ball(y, half_window=half_window)
                y = y - y_base
                
            # Suavizado
            if smoothing_algo == "savgol":
                window_size = 11 if len(y) >= 11 else (len(y) // 2 * 2 + 1)
                if window_size >= 3:
                    y = signal.savgol_filter(y, window_length=window_size, polyorder=2)
            elif smoothing_algo == "movingavg":
                kernel = np.ones(5) / 5.0
                y = np.convolve(y, kernel, mode='same')
                
            processed_results.append({
                "name": request.spectra[idx].name,
                "x": x.tolist(),
                "y": y.tolist()
            })
            
        return {"spectra": processed_results}
    except Exception as e:
        error_trace = traceback.format_exc()
        print(error_trace) 
        return JSONResponse(status_code=500, content={"detail": f"Error matemático en Pipeline: {str(e)}"})

@app.post("/comparar")
async def comparar_espectros_avanzado(data: CompareRequest):
    if len(data.spectra) < 2:
        return {"error": "Se requieren al menos 2 espectros para comparar."}
    
    spectra_peaks = []
    for spec in data.spectra:
        y = np.array([v if v is not None else 0.0 for v in spec.y])
        x = np.array([v if v is not None else 0.0 for v in spec.x])
        prominence = (np.max(y) - np.min(y)) * 0.05
        peak_idx, _ = signal.find_peaks(y, prominence=prominence)
        spectra_peaks.append([{"x": float(x[i]), "y": float(y[i])} for i in peak_idx])
        
    diff_peaks_result = {}
    for i, spec in enumerate(data.spectra):
        unique_peaks = []
        for p1 in spectra_peaks[i]:
            diff_type = None
            is_diff = False
            for j in range(len(data.spectra)):
                if i == j: continue
                matching_peaks = [p2 for p2 in spectra_peaks[j] if abs(p1["x"] - p2["x"]) <= 4.0]
                if not matching_peaks:
                    diff_type = "Pico Diferencial"
                    is_diff = True
                    break
                else:
                    closest_peak = min(matching_peaks, key=lambda p2: abs(p1["x"] - p2["x"]))
                    intensity_diff = abs(p1["y"] - closest_peak["y"]) / max(abs(p1["y"]), 1e-9)
                    if intensity_diff > 0.25:
                        diff_type = "Cambio de Intensidad"
                        is_diff = True
                        break
            if is_diff:
                unique_peaks.append({"x": p1["x"], "y": p1["y"], "type": diff_type})
        diff_peaks_result[spec.name] = unique_peaks
        
    return {"diff_peaks": diff_peaks_result}


def build_symmetric_matrix(data: list[SpectrumInput]):
    """
    Construye una matriz de datos alineada mediante interpolación en el rango común.
    Garantiza ordenamiento monotónico para Scipy/Numpy.
    """
    all_x = []
    for s in data:
        x = np.array([v if v is not None else 0.0 for v in s.x])
        all_x.append(x)
    
    # 1. Rango común estricto (Intersección Global)
    global_min = max([np.min(x) for x in all_x])
    global_max = min([np.max(x) for x in all_x])
    
    if global_min >= global_max:
        raise ValueError("No hay un rango común (intersección) entre los espectros seleccionados.")
        
    # 2. Eje X Maestro solo en el rango seguro
    x_ref = np.arange(global_min, global_max, 1.0)
    
    Y_list = []
    for s in data:
        # x_temp y y_temp son los arrays extraídos
        x_temp = np.array([v if v is not None else 0.0 for v in s.x], dtype=float)
        y_temp = np.array([v if v is not None else 0.0 for v in s.y], dtype=float)
        y_temp = np.nan_to_num(y_temp, nan=0.0)
        
        # Ordenamiento monotónico de menor a mayor obligatorio para Scipy
        sort_idx = np.argsort(x_temp)
        x_temp = x_temp[sort_idx]
        y_temp = y_temp[sort_idx]
        
        # Aplicación de interpolación Scipy
        f_interp = interp1d(x_temp, y_temp, kind='linear', fill_value='extrapolate', bounds_error=False)
        y_interp = f_interp(x_ref)
        Y_list.append(y_interp)
        
    return np.array(Y_list), x_ref

def prepare_chemometric_matrix(data: ChemoRequest):
    """
    Aplica el Pipeline Quimiométrico: Alineación -> Recorte -> Escalado.
    """
    Y_matrix, x_ref = build_symmetric_matrix(data.spectra)
    df = pd.DataFrame(Y_matrix, columns=x_ref.astype(float))
    
    # 1. RECORTE (Spectral Range)
    r_min, r_max = data.params.range[0], data.params.range[1]
    mask = (df.columns >= r_min) & (df.columns <= r_max)
    df = df.loc[:, mask]
    
    if df.empty:
        raise ValueError(f"El rango [{r_min}, {r_max}] no contiene datos.")
    
    # 2. ESCALADO (Normalización Dinámica)
    Y = df.values
    if data.params.scale == "minmax":
        from sklearn.preprocessing import MinMaxScaler
        Y = MinMaxScaler().fit_transform(Y.T).T 
    elif data.params.scale == "snv":
        mean = np.mean(Y, axis=1, keepdims=True)
        std = np.std(Y, axis=1, keepdims=True)
        Y = (Y - mean) / (std + 1e-9)
        
    return Y, df.columns.values

@app.post("/api/pca")
async def calculate_pca(data: ChemoRequest):
    try:
        if len(data.spectra) < 2: return {"error": "Se requieren al menos 2 espectros."}
        names = [s.name for s in data.spectra]
        Y, _ = prepare_chemometric_matrix(data)
        
        n_comps = min(2, Y.shape[0])
        pca = PCA(n_components=n_comps)
        scores = pca.fit_transform(Y)
        evr = pca.explained_variance_ratio_ * 100
        
        scores_out = []
        for i, n in enumerate(names):
            pc1 = float(scores[i][0])
            pc2 = float(scores[i][1]) if n_comps > 1 else 0.0
            scores_out.append({"name": n, "pc1": pc1, "pc2": pc2})
            
        return {
            "type": "pca",
            "scores": scores_out,
            "explained_variance": [float(evr[0]), float(evr[1]) if n_comps > 1 else 0.0]
        }
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"detail": f"Error matemático PCA: {str(e)}"})

@app.post("/api/hca")
async def calculate_hca(data: ChemoRequest):
    try:
        if len(data.spectra) < 2: return {"error": "Se requieren al menos 2 espectros."}
        names = [s.name for s in data.spectra]
        
        # 1. Limpieza automática de etiquetas para legibilidad
        clean_labels = [
            str(name).replace('.csv', '')
                     .replace('.txt', '')
                     .replace('CONVERTED_T2A_', '')
                     .replace('prom_', '')
            for name in names
        ]
        
        Y, _ = prepare_chemometric_matrix(data)
        
        Z = linkage(Y, method=data.linkage_method, metric='euclidean')
        
        # 2. Generación del dendrograma sin truncamiento
        ddata = dendrogram(
            Z, 
            labels=clean_labels, 
            no_plot=True,
            truncate_mode=None, 
            color_threshold=data.color_threshold, # Umbral dinámico
            leaf_rotation=90.,
            leaf_font_size=10.,
            show_contracted=False
        )
        
        return {
            "type": "hca",
            "icoord": ddata['icoord'],
            "dcoord": ddata['dcoord'],
            "ivl": ddata['ivl']
        }
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"detail": f"Error matemático HCA: {str(e)}"})

@app.post("/api/correlation")
async def calculate_correlation(data: ChemoRequest):
    try:
        if len(data.spectra) < 2: return {"error": "Se requieren al menos 2 espectros."}
        names = [s.name for s in data.spectra]
        Y, _ = prepare_chemometric_matrix(data)
        
        corr_matrix = np.corrcoef(Y)
        return {
            "type": "correlation",
            "matrix": corr_matrix.tolist(),
            "labels": names
        }
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"detail": f"Error matemático Correlación: {str(e)}"})

@app.post("/api/pls_da")
async def calculate_plsda(data: PlsdaRequest):
    try:
        if len(data.spectra) < 3: return {"error": "Se requieren al menos 3 espectros para entrenar PLS-DA."}
        
        names = [s.name for s in data.spectra]
        labels_raw = [s.label for s in data.spectra]
        
        Y_features, x_ref = build_symmetric_matrix(data.spectra)
        
        # Binarización One-Hot de etiquetas de texto
        le = LabelBinarizer()
        Y_target = le.fit_transform(labels_raw)
        
        # Ajuste dinámico de tensores PLS
        n_comps = max(1, min(data.n_components, Y_features.shape[0]-1))
        
        pls = PLSRegression(n_components=n_comps)
        pls.fit(Y_features, Y_target)
        scores = pls.x_scores_
        
        # Pesos espectrales (Biomarcadores predictivos) usando matriz coef_ paramétrica abstracta
        if pls.coef_.ndim > 1:
            loadings = np.mean(np.abs(pls.coef_), axis=1)
        else:
            loadings = np.abs(pls.coef_).flatten()
            
        scores_grouped = {}
        for i, grp in enumerate(labels_raw):
            if grp not in scores_grouped:
                scores_grouped[grp] = []
            scores_grouped[grp].append({
                "name": names[i],
                "lv1": float(scores[i, 0]) if n_comps > 0 else 0.0,
                "lv2": float(scores[i, 1]) if n_comps > 1 else 0.0
            })
            
        return {
            "scores": scores_grouped,
            "vip": {"x": x_ref.tolist(), "y": loadings.tolist()}
        }
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"detail": f"Error matemático PLS-DA: {str(e)}"})
