from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
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

app = FastAPI(title="Merschel-Raman V8.2 API")

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

class ChemoRequest(BaseModel):
    spectra: List[SpectrumData]
    analysis_type: str # "pca", "hca", "correlation"

class CompareRequest(BaseModel):
    spectra: List[SpectrumData]

# Ruta para servir nuestra UI Front-end
@app.get("/", response_class=HTMLResponse)
async def read_index():
    return FileResponse("../public/index.html")

# Endpoint matemático y de procesamiento quimiométrico
@app.post("/api/process")
async def process_spectra(request: ProcessRequest):
    try:
        baseline_algo = request.config.baseline
        smoothing_algo = request.config.smoothing
        
        baseline_fitter = Baseline()
        processed_results = []
        
        for spec in request.spectra:
            y = np.array([v if v is not None else 0.0 for v in spec.y])
            x = np.array([v if v is not None else 0.0 for v in spec.x])
            
            # 1. Corrección de Línea Base
            if baseline_algo == "als":
                y_base, _ = baseline_fitter.asls(y, lam=1e5, p=0.01)
                y = y - y_base
            elif baseline_algo == "rollingball":
                half_window = max(5, len(y) // 20)
                y_base, _ = baseline_fitter.rolling_ball(y, half_window=half_window)
                y = y - y_base
                
            # 2. Suavizado
            if smoothing_algo == "savgol":
                window_size = 11 if len(y) >= 11 else (len(y) // 2 * 2 + 1)
                if window_size >= 3:
                    y = signal.savgol_filter(y, window_length=window_size, polyorder=2)
            elif smoothing_algo == "movingavg":
                kernel = np.ones(5) / 5.0
                y = np.convolve(y, kernel, mode='same')
                
            processed_results.append({
                "name": spec.name,
                "x": x.tolist(),
                "y": y.tolist()
            })
            
        return {"spectra": processed_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fallo en procesamiento espectral: {str(e)}")

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


# Interpolación genérica (Helper Function)
def build_symmetric_matrix(data: list[SpectrumInput]):
    x_ref = np.array([v if v is not None else 0.0 for v in data[0].x])
    Y_list = []
    
    for s in data:
        x_s = np.array([v if v is not None else 0.0 for v in s.x])
        y_s = np.array([v if v is not None else 0.0 for v in s.y])
        y_s = np.nan_to_num(y_s, nan=0.0, posinf=0.0, neginf=0.0)
        
        if not np.array_equal(x_ref, x_s):
            Y_list.append(np.interp(x_ref, x_s, y_s))
        else:
            Y_list.append(y_s)
    return np.array(Y_list)

@app.post("/api/pca")
async def calculate_pca(data: list[SpectrumInput]):
    try:
        if len(data) < 2: return {"error": "Se requieren al menos 2 espectros."}
        names = [s.name for s in data]
        Y = build_symmetric_matrix(data)
        
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
        raise HTTPException(status_code=500, detail=f"Fallo en análisis PCA: {str(e)}")

@app.post("/api/hca")
async def calculate_hca(data: list[SpectrumInput]):
    try:
        if len(data) < 2: return {"error": "Se requieren al menos 2 espectros."}
        names = [s.name for s in data]
        Y = build_symmetric_matrix(data)
        
        Z = linkage(Y, method='ward', metric='euclidean')
        ddata = dendrogram(Z, labels=names, no_plot=True)
        return {
            "type": "hca",
            "icoord": ddata['icoord'],
            "dcoord": ddata['dcoord'],
            "ivl": ddata['ivl']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fallo en análisis HCA: {str(e)}")

@app.post("/api/correlation")
async def calculate_correlation(data: list[SpectrumInput]):
    try:
        if len(data) < 2: return {"error": "Se requieren al menos 2 espectros."}
        names = [s.name for s in data]
        Y = build_symmetric_matrix(data)
        
        corr_matrix = np.corrcoef(Y)
        return {
            "type": "correlation",
            "matrix": corr_matrix.tolist(),
            "labels": names
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fallo en cálculo de correlación: {str(e)}")

@app.post("/api/pls_da")
async def calculate_plsda(data: PlsdaRequest):
    try:
        if len(data.spectra) < 3: return {"error": "Se requieren al menos 3 espectros para entrenar PLS-DA."}
        
        names = [s.name for s in data.spectra]
        labels_raw = [s.label for s in data.spectra]
        
        Y_features = build_symmetric_matrix(data.spectra)
        x_ref = np.array([v if v is not None else 0.0 for v in data.spectra[0].x])
        
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
        raise HTTPException(status_code=500, detail=f"Fallo en análisis PLS-DA: {str(e)}")
