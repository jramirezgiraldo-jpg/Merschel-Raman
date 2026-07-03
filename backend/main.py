from fastapi import FastAPI, HTTPException, File, Form, UploadFile
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from scipy import signal
from pybaselines import Baseline
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

class PLSExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.pls = PLSRegression(n_components=self.n_components)
        self.lb = LabelBinarizer()

    def fit(self, X, y):
        # Binarización ortogonal para evitar regresión ordinal
        y_bin = self.lb.fit_transform(y)
        self.pls.fit(X, y_bin)
        return self

    def transform(self, X):
        # Extrae exclusivamente los scores latentes limpios (X_scores)
        return self.pls.transform(X)

from fastapi.middleware.cors import CORSMiddleware
import re
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, savgol_filter
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve
from joblib import Parallel, delayed
import hashlib
import os
import io
import struct

def parse_raw_spectroscopy_file(file_content: bytes, filename: str):
    """
    Extractor Universal Nativo: Parsea archivos binarios .sp (PerkinElmer PEPE2D) 
    y .csv estructurados, extrayendo las mallas vectoriales.
    """
    if filename.lower().endswith('.sp'):
        try:
            if b'PEPE2D' in file_content[:20]:
                if len(file_content) >= 12408:
                    floats = struct.unpack('<1551d', file_content[-12408:])
                    wavenumbers = np.linspace(4000, 900, 1551)
                    return wavenumbers, np.array(floats)
        except Exception as e:
            raise ValueError(f"Error parseando .sp binario: {e}")
            
    try:
        text = file_content.decode('utf-8', errors='ignore')
        lines = text.splitlines()
        
        # Buscar la fila que contiene 'wavenumber' y 'absorbance'
        header_idx = 0
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if 'wavenumber' in line_lower or 'absorbance' in line_lower:
                header_idx = i
                break
                
        # Parsear ignorando las filas previas al header detectado
        df = pd.read_csv(io.StringIO(text), skiprows=header_idx)
        
        # Normalizar cabeceras a minúsculas y buscar las columnas relevantes
        df.columns = [str(c).strip().lower() for c in df.columns]
        
        col_wav = next((c for c in df.columns if 'wavenumber' in c or 'cm-1' in c or 'x' in c), None)
        col_abs = next((c for c in df.columns if 'absorbance' in c or 'y' in c or 'intensit' in c), None)
        
        if col_wav and col_abs:
            df['wavenumber'] = pd.to_numeric(df[col_wav], errors='coerce')
            df['absorbance'] = pd.to_numeric(df[col_abs], errors='coerce')
            df = df.dropna(subset=['wavenumber', 'absorbance'])
            if len(df) > 10:
                df = df.sort_values(by='wavenumber', ascending=True).reset_index(drop=True)
                return df['wavenumber'].values, df['absorbance'].values
    except Exception:
        pass
        
    text = file_content.decode('utf-8', errors='ignore')
    lines = text.splitlines()
    cleaned_data = []
    for line in lines:
        parts = re.split(r'[,\t;]+|\s{2,}', line.strip())
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) >= 2:
            try:
                cleaned_data.append([float(parts[0]), float(parts[1])])
            except ValueError:
                pass
                
    if not cleaned_data:
        raise ValueError("No se encontraron datos numéricos válidos en el archivo.")
        
    df = pd.DataFrame(cleaned_data, columns=['Wavenumber', 'Absorbance'])
    df = df.sort_values(by='Wavenumber', ascending=True).reset_index(drop=True)
    return df['Wavenumber'].values, df['Absorbance'].values

def aplicar_quimiometria(espectros_json):
    """
    Procesador altamente eficiente mediante operaciones vectorizadas.
    Usa pre-asignación de memoria para manejar miles de espectros.
    """
    n_espectros = len(espectros_json)
    malla_fija = np.linspace(900.0, 4000.0, 1550)
    X_mat = np.zeros((n_espectros, 1550))
    y_list = []
    
    for i, item in enumerate(espectros_json):
        x_raw, y_raw = np.array(item["wavenumbers"], dtype=float), np.array(item["absorbances"], dtype=float)
        
        # Alineación matemática (evitando copias innecesarias)
        idx = np.argsort(x_raw)
        f = interp1d(x_raw[idx], y_raw[idx], kind='linear', bounds_error=False, fill_value="extrapolate")
        X_mat[i, :] = np.nan_to_num(f(malla_fija), nan=0.0)
        y_list.append(str(item["label"]).strip())
        
    # Normalización SNV vectorizada para todo el conjunto
    X_mat = (X_mat - X_mat.mean(axis=1, keepdims=True)) / (X_mat.std(axis=1, keepdims=True) + 1e-8)
    
    # Derivada de Savitzky-Golay
    X_deriv = savgol_filter(X_mat, window_length=15, polyorder=2, deriv=1, axis=1)
    
    return X_deriv, np.array(y_list)

app = FastAPI(title="Hershell-Raman V8.2 API")

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(status_code=400, content={"detail": str(exc.errors())})

# El middleware DEBE ir inmediatamente después de la instanciación de app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos de datos para el Body de la petición (JSON)
def clean_sample_name(name: str):
    """
    Purga forzada: elimina cualquier rastro de HTML o extensión de forma agresiva.
    """
    nombre = str(name)
    # 1. Eliminar explícitamente el bug "i>" y cualquier tag
    nombre = nombre.replace('i>', '').replace('<i>', '').replace('</i>', '')
    
    # 2. PURGA REGEX: Mantener SOLO letras, números, espacios y guiones
    import re
    nombre = re.sub(r'[^a-zA-Z0-9\s\-]', '', nombre)
    
    # 3. Limpiar extensiones y residuos
    nombre = nombre.replace('csv', '').replace('txt', '').replace('asc', '').strip()
    
    # 4. Fallback seguro
    if not nombre or nombre.strip() == "":
        nombre = "Espectro Recuperado"
        
    return nombre.strip()

def get_clean_italic_name(filename: str) -> str:
    """
    Función canónica de nomenclatura científica (espejo del get_clean_italic_name() de JS).
    SIEMPRE devuelve "Nombre Limpio" garantizando que NO haya HTML.
    a) Elimina extensiones (.csv, .txt, .asc, .dat) via os.path.splitext
    b) Limpia guiones bajos y términos técnicos (Raman, FTIR, raw, proc)
    """
    clean = clean_sample_name(filename)
    # Garantía de contenido: si el nombre quedó vacío, usar fallback
    if not clean or not clean.strip() or clean == "i>":
        clean = "Espectro Desconocido"
    return clean

def format_scientific_name(name: str, use_latex: bool = False) -> str:
    """
    Alias de get_clean_italic_name() para compatibilidad con código existente.
    Soporta formato LaTeX (Matplotlib) y HTML (Plotly).
    """
    clean = clean_sample_name(name)
    if not clean or not clean.strip() or clean == "i>":
        clean = "Espectro Desconocido"
    if use_latex:
        return f"$\\mathit{{{clean}}}$"
    return clean

# --- CACHE DE PROCESAMIENTO ---
_process_cache = {}

def get_payload_hash(payload_dict):
    import json
    return hashlib.md5(json.dumps(payload_dict, sort_keys=True, default=str).encode()).hexdigest()

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

class PredictRequest(BaseModel):
    train_spectra: list[LabeledSpectrumInput]
    test_spectra: list[SpectrumInput]
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

class CharacterizeRequest(BaseModel):
    spectra: list[SpectrumInput]
    prominence: float = 0.05
    method: str = "Raman"
    tolerance: float = 5.0

# Ruta para servir nuestra UI Front-end
@app.get("/", response_class=HTMLResponse)
async def read_index():
    return FileResponse("../public/index.html")

@app.post("/api/process")
async def process_spectra(request: ProcessRequest):
    try:
        # 1. Caching Check
        cache_key = get_payload_hash(request.dict())
        if cache_key in _process_cache:
            return _process_cache[cache_key]

        baseline_algo = request.config.baseline
        smoothing_algo = request.config.smoothing
        
        Y_matrix, x_ref = build_symmetric_matrix(request.spectra)
        
        # 1. OPTIMIZACIÓN EXTREMA VECTORIZADA (Suavizado Matricial)
        if smoothing_algo == "savgol":
            window_size = 11 if Y_matrix.shape[1] >= 11 else (Y_matrix.shape[1] // 2 * 2 + 1)
            if window_size >= 3:
                Y_matrix = signal.savgol_filter(Y_matrix, window_length=window_size, polyorder=2, axis=1)
        elif smoothing_algo == "movingavg":
            # Convolución 2D Vectorizada a toda la matriz
            kernel = np.ones((1, 5)) / 5.0
            Y_matrix = signal.convolve2d(Y_matrix, kernel, mode='same')
            
        # 2. OPTIMIZACIÓN EXTREMA (Corrección de Línea Base en Paralelo con max_iter)
        baseline_fitter = Baseline()
        if baseline_algo == "als":
            def _base_als(y):
                base, _ = baseline_fitter.asls(y, lam=1e5, p=0.01, max_iter=10) # max_iter limitado para evitar timeouts
                return y - base
            Y_matrix = np.array(Parallel(n_jobs=-1)(delayed(_base_als)(y) for y in Y_matrix))
        elif baseline_algo == "rollingball":
            half_window = max(5, Y_matrix.shape[1] // 20)
            def _base_rb(y):
                base, _ = baseline_fitter.rolling_ball(y, half_window=half_window)
                return y - base
            Y_matrix = np.array(Parallel(n_jobs=-1)(delayed(_base_rb)(y) for y in Y_matrix))

        # 3. Empaquetar
        from functools import lru_cache
        processed_results = []
        for i in range(len(Y_matrix)):
            processed_results.append({
                # CRÍTICO: Devolver nombre limpio SIN tags HTML.
                "name": clean_sample_name(request.spectra[i].name),
                "x": x_ref.tolist(),
                "y": Y_matrix[i].tolist()
            })
        
        res = {"spectra": processed_results}
        _process_cache[cache_key] = res
        return res
    except Exception as e:
        import traceback
        print(traceback.format_exc()) 
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
        # Usar nombre limpio sin HTML para claves del diccionario JSON
        diff_peaks_result[clean_sample_name(spec.name)] = unique_peaks
        
    return {"diff_peaks": diff_peaks_result}

@app.post("/api/characterize")
async def characterize_spectra(request: CharacterizeRequest):
    try:
        if not request.spectra:
            return {"error": "No hay espectros para caracterizar."}
            
        # 1. Alineación y Matrix Builder
        Y_matrix, x_ref = build_symmetric_matrix(request.spectra)
        # clean_sample_name: texto plano, sin formato HTML.
        names = [clean_sample_name(s.name) for s in request.spectra]
        
        all_peaks_x = []
        spectra_peak_details = [] # List of dicts per spectrum
        
        # 2. Detección individual para encontrar anchos y prominencias reales
        for idx, y in enumerate(Y_matrix):
            # Normalización local para find_peaks si es necesario, 
            # pero usaremos la prominencia absoluta escalada por el rango del espectro
            local_range = np.max(y) - np.min(y)
            prom = request.prominence * local_range if local_range > 0 else 0.1
            
            peaks, props = signal.find_peaks(y, prominence=prom, width=True)
            
            current_spec_peaks = []
            for i, p_idx in enumerate(peaks):
                wn = float(x_ref[p_idx])
                inte = float(y[p_idx])
                width = float(props['widths'][i]) # FWHM aproximado por scipy
                current_spec_peaks.append({"x": wn, "y": inte, "width": width})
                all_peaks_x.append(wn)
            
            spectra_peak_details.append(current_spec_peaks)
            
        if not all_peaks_x:
            return {"peaks": [], "table": []}
            
        # 3. Consolidación de picos (Clustering por cercanía)
        all_peaks_x.sort()
        groups = []
        if all_peaks_x:
            current_group = [all_peaks_x[0]]
            for x in all_peaks_x[1:]:
                if x - current_group[-1] <= 6.0: # Tolerancia de 6 cm-1
                    current_group.append(x)
                else:
                    groups.append(np.mean(current_group))
                    current_group = [x]
            groups.append(np.mean(current_group))
            
        # 4. Generación de Tabla y Asignaciones
        def get_assignment(wn):
            # DICCIONARIO ROBUSTO CIENTÍFICO (800-1800 cm-1)
            # 1. Picos ultra-específicos (Prioridad máxima)
            if 1650 <= wn <= 1675: return "Estiramiento C=O: Amida I (Proteínas)"
            if 1540 <= wn <= 1560: return "Deformación N-H: Amida II (Proteínas)"
            if 1070 <= wn <= 1090: return "Estiramiento C-O: Enlaces Glicosídicos (Carbohidratos)"
            if 1003 <= wn <= 1005: return "Respiración del anillo: Fenilalanina (Proteínas)"
            
            # 2. Rangos principales y modos vibracionales anchos (Nuevos requerimientos)
            if 1400 <= wn <= 1480: return "Deformación C-H2 / C-H3 (Scissoring/Flexión): Lípidos / Cadenas alifáticas"
            if 1300 <= wn < 1400: return "Deformación C-H (Torsión/Wagging): Colágeno / Proteínas"
            if 1200 <= wn < 1300: return "Estiramiento asimétrico P=O y Deformación N-H: Ácidos Nucleicos / Amida III"
            if 1150 <= wn <= 1180: return "Estiramiento asimétrico C-O-C: Carbohidratos / Enlaces Glicosídicos"
            if 1000 <= wn <= 1050: return "Estiramiento C-O y C-N: Glucógeno / Residuos de aminoácidos"
            if 900 <= wn <= 950: return "Estiramiento C-C y Deformación C-H: Carbohidratos/Proteínas"
            
            # 3. Regla del Modo Predominante para valles o "zonas muertas" (Sin fallbacks ambiguos)
            if wn > 1560: return "Deformación N-H y Estiramiento C=O: Interacción de Proteínas"
            if 1480 < wn < 1540: return "Deformación C-H y Estiramiento C-N: Amida II mixta (Proteínas)"
            if 1090 < wn < 1150: return "Tensión de esqueleto C-C y C-O: Carbohidratos complejos"
            if 950 < wn < 1000: return "Vibración de estiramiento esquelético C-C: Modo alifático"
            if wn < 900: return "Tensión de esqueleto C-C y deformaciones anulares: Carbohidratos"
            
            return "Tensión de esqueleto vibracional mixta: Interacción Fingerprint"

        table_data = []
        for g_wn in groups:
            row = {
                "wavenumber": round(g_wn, 1),
                "assignment": get_assignment(g_wn)
            }
            # Encontrar intensidades para cada muestra
            for s_idx, name in enumerate(names):
                # Buscar si hay un pico en este grupo para esta muestra
                match = next((p for p in spectra_peak_details[s_idx] if abs(p["x"] - g_wn) <= 5.0), None)
                if match:
                    row[name] = round(match["y"], 4)
                    row[f"{name}_width"] = round(match["width"], 2)
                else:
                    # Si no hay pico, tomamos el valor interpolado de la matriz
                    idx_nearest = np.abs(x_ref - g_wn).argmin()
                    row[name] = round(float(Y_matrix[s_idx, idx_nearest]), 4)
                    row[f"{name}_width"] = None
                    
            table_data.append(row)
            
        return {
            "peaks": spectra_peak_details, 
            "table": table_data,
            "x_ref": x_ref.tolist(),
            "y_matrix": Y_matrix.tolist(),
            "names": names
        }
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.post("/api/report")
async def generate_taxonomic_report(request: CharacterizeRequest):
    try:
        method = request.method.upper()
        # 1. Alineación y Matrix Builder
        Y_matrix_full, x_ref_full = build_symmetric_matrix(request.spectra)
        
        # 2. Recorte forzado a la región Fingerprint (800 - 1800 cm⁻¹)
        mask = (x_ref_full >= 800) & (x_ref_full <= 1800)
        x_ref = x_ref_full[mask]
        Y_matrix_raw = Y_matrix_full[:, mask]
        
        # Agrupación Taxonómica Automática (Orden Alfabético de Especies)
        orig_names = [s.name for s in request.spectra]
        sorted_indices = sorted(range(len(orig_names)), key=lambda i: orig_names[i])
        
        # Nombres formateados para reportes HTML y Plotly
        names = [format_scientific_name(orig_names[i]) for i in sorted_indices]
        Y_matrix = Y_matrix_raw[sorted_indices]
        n_samples = len(names)
        
        all_peaks_x = []
        spectra_peak_details = []
        
        for idx, y in enumerate(Y_matrix):
            local_range = np.max(y) - np.min(y)
            prom = request.prominence * local_range if local_range > 0 else 0.1
            peaks, _ = signal.find_peaks(y, prominence=prom)
            current_spec_peaks = []
            for p_idx in peaks:
                wn = float(x_ref[p_idx])
                current_spec_peaks.append({"x": wn, "y": float(y[p_idx])})
                all_peaks_x.append(wn)
            spectra_peak_details.append(current_spec_peaks)
            
        all_peaks_x.sort()
        groups = []
        if all_peaks_x:
            current_group = [all_peaks_x[0]]
            for x in all_peaks_x[1:]:
                if x - current_group[0] <= request.tolerance: # Limitar el ancho total del clúster dinámicamente
                    current_group.append(x)
                else:
                    groups.append(np.mean(current_group))
                    current_group = [x]
            groups.append(np.mean(current_group))
            
        def get_assignment(wn, meth):
            # TÉCNICA ESPECÍFICA (FTIR / RAMAN)
            if 1003 <= wn <= 1005: return "Respiración del anillo: Fenilalanina (Proteínas)"
            if 1650 <= wn <= 1675: return "Estiramiento C=O: Amida I (Proteínas)"
            if 1440 <= wn <= 1455: return "Deformación CH2: Flexión (Lípidos)"
            if 1070 <= wn <= 1090: return "Estiramiento C-O: Enlaces Glicosídicos (Carbohidratos)"
            if 1240 <= wn <= 1265: return "Estiramiento P=O asimétrico: Fosfodiéster (Ácidos Nucleicos)"
            if 1735 <= wn <= 1745: return "Estiramiento C=O: Ésteres (Lípidos/FTIR)"
            if 930 <= wn <= 950:   return "Vibración C-C: Esqueleto proteico (α-hélice)"
            
            # CATEGORÍAS POR REGIÓN (FALLBACK SIN 'BIOQUÍMICA')
            if 1500 <= wn <= 1800: return "Señal compleja en región de Proteínas"
            if 1200 <= wn < 1500: return "Señal compleja en región de Lípidos / Ácidos Nucleicos"
            if 900 <= wn < 1200: return "Señal compleja en región de Carbohidratos / ADN"
            
            return "Señal vibracional identificada en zona Fingerprint"

        # Clasificación y Recolección
        common_rows = []
        shared_rows = []
        unique_by_sample = {n: [] for n in names}

        for g_wn in groups:
            present_in = []
            wn_values = []
            for s_idx, name in enumerate(names):
                match = next((p for p in spectra_peak_details[s_idx] if abs(p["x"] - g_wn) <= request.tolerance), None)
                if match:
                    present_in.append(name)
                    wn_values.append(f"{match['x']:.1f}")
                else:
                    wn_values.append("-")
            
            count = len(present_in)
            assignment = get_assignment(g_wn, method)
            sample_cols = "".join([f"<td>{v}</td>" for v in wn_values])
            
            if count == n_samples:
                common_rows.append(f"""
                <tr class="row-common">
                    <td class="char-text-common"><b>COINCIDENCIA COMÚN (100%)</b></td>
                    <td>{assignment}</td>
                    {sample_cols}
                </tr>""")
            elif count > 1:
                shared_rows.append(f"""
                <tr class="row-shared">
                    <td class="char-text-shared"><b>Compartido ({' y '.join(present_in)})</b></td>
                    <td>{assignment}</td>
                    {sample_cols}
                </tr>""")
            else:
                unique_owner = present_in[0]
                unique_by_sample[unique_owner].append(f"""
                <tr class="row-unique">
                    <td class="char-text-unique"><b>Diferenciador Único ({unique_owner})</b></td>
                    <td>{assignment}</td>
                    {sample_cols}
                </tr>""")

        # Ensamblar filas únicas agrupadas
        unique_rows_compiled = []
        for n in names:
            if unique_by_sample[n]:
                unique_rows_compiled.append(f'<tr class="row-group-header"><td colspan="{n_samples + 2}"><b>Biomarcadores Únicos de: {n}</b></td></tr>')
                unique_rows_compiled.extend(unique_by_sample[n])

        headers_samples = "".join([f"<th>{n}</th>" for n in names])
        
        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Caracterización Profesional Hershell Raman</title>
            <style>
                body {{ font-family: 'Open Sans', Arial, sans-serif; background: #fdfdfd; color: #333; padding: 40px; line-height: 1.6; margin-bottom: 100px; }}
                .main-container {{ margin-left: 240px; }}
                .report-card {{ background: #fff; padding: 40px; border-radius: 4px; box-shadow: 0 10px 25px rgba(0,0,0,0.05); overflow: visible; }}
                .brand-header {{ text-align: right; font-size: 0.7rem; color: #aaa; font-weight: bold; margin-bottom: 20px; text-transform: uppercase; letter-spacing: 2px; }}
                h1 {{ color: #00796b; text-align: center; font-weight: 800; border-bottom: 4px solid #009b77; padding-bottom: 5px; margin-bottom: 5px; margin-top: 0; }}
                .subtitle {{ text-align: center; color: #666; font-size: 0.9rem; margin-bottom: 30px; font-style: italic; }}
                
                /* PANEL CONSOLA LATERAL IZQUIERDA */
                .sidebar-console {{
                    position: fixed; left: 15px; top: 40px; width: 220px; z-index: 9999;
                    display: flex; flex-direction: column; gap: 20px;
                }}
                .sidebar-group {{ background: #fff; border: 1px solid #ddd; border-radius: 10px; padding: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
                .sidebar-label {{ font-size: 0.65rem; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; font-weight: bold; display: block; }}
                
                .btn-list {{ display: flex; flex-direction: column; gap: 8px; }}
                .filter-btn {{
                    padding: 10px; background: #f8fafc; border: 1px solid #eee; border-radius: 6px;
                    cursor: pointer; font-size: 0.7rem; font-weight: bold; text-align: left;
                    transition: all 0.2s;
                }}
                .filter-btn:hover {{ background: #f1f5f9; border-color: #009b77; }}
                .filter-btn.active {{ background: #009b77; color: white; border-color: #009b77; }}

                .legend-box {{ display: flex; flex-direction: column; gap: 8px; }}
                .l-item {{ display: flex; align-items: center; gap: 10px; font-size: 0.7rem; font-weight: bold; color: #444; }}
                .box {{ width: 14px; height: 14px; border-radius: 2px; border: 1px solid rgba(0,0,0,0.1); }}

                /* TABLA Y ESTILOS */
                table {{ 
                    width: 100%; 
                    border-collapse: separate; 
                    border-spacing: 0; 
                    margin-top: 10px; 
                }}
                thead th {{ 
                    position: -webkit-sticky;
                    position: sticky; 
                    top: 0; 
                    z-index: 999;
                    background-color: #009b77 !important; 
                    color: #fff; 
                    padding: 15px; 
                    text-align: left; 
                    font-size: 0.8rem; 
                    text-transform: uppercase; 
                    border-bottom: 2px solid #333;
                }}
                tbody td {{ padding: 12px; border: 1px solid #eee; font-size: 0.85rem; }}
                .row-group-header {{ background: #f8fafc; color: #64748b; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; }}
                .row-common {{ background-color: #f1fcf9; }}
                .row-shared {{ background-color: #f0f7ff; }}
                .row-unique {{ background-color: #fff9f0; }}
                .char-text-common {{ color: #00796b; }}
                .char-text-shared {{ color: #1e40af; }}
                .char-text-unique {{ color: #92400e; }}
                .footer {{ margin-top: 50px; text-align: center; font-size: 0.7rem; color: #bbb; }}
            </style>
            <script>
                function filterRows(type, btn) {{
                    const rows = document.querySelectorAll('tbody tr');
                    rows.forEach(r => {{
                        if (r.classList.contains('row-group-header')) {{
                            r.style.display = (type === 'all' || type === 'row-unique') ? '' : 'none';
                            return;
                        }}
                        if (type === 'all') r.style.display = '';
                        else if (r.classList.contains(type)) r.style.display = '';
                        else r.style.display = 'none';
                    }});
                    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                }}
            </script>
        </head>
        <body>
            <div class="sidebar-console">
                <div class="sidebar-group">
                    <span class="sidebar-label">Filtros de Análisis</span>
                    <div class="btn-list">
                        <button class="filter-btn active" onclick="filterRows('all', this)">Ver Tabla Completa</button>
                        <button class="filter-btn" onclick="filterRows('row-common', this)">Coincidencias Totales</button>
                        <button class="filter-btn" onclick="filterRows('row-shared', this)">Picos Compartidos</button>
                        <button class="filter-btn" onclick="filterRows('row-unique', this)">Diferenciadores Únicos</button>
                    </div>
                </div>
                
                <div class="sidebar-group">
                    <span class="sidebar-label">Categorización ({method})</span>
                    <div class="legend-box">
                        <div class="l-item"><div class="box" style="background:#f1fcf9; border-color:#009b77;"></div> COMÚN (100%)</div>
                        <div class="l-item"><div class="box" style="background:#f0f7ff; border-color:#1e40af;"></div> COMPARTIDO</div>
                        <div class="l-item"><div class="box" style="background:#fff9f0; border-color:#92400e;"></div> ÚNICO (Biomarcador)</div>
                    </div>
                </div>
            </div>

            <div class="main-container">
                <div class="report-card">
                    <div class="brand-header">Hershell-Raman | Scientific Publication Standard</div>
                    <h1>REPORTE TAXONÓMICO {method}</h1>
                    <div class="subtitle">Análisis de Marcadores Bioquímicos en la Región Fingerprint (800 - 1800 cm⁻¹)</div>
                    
                    <table class="table-multi-spectrum">
                        <thead>
                            <tr>
                                <th>CARACTERÍSTICAS</th>
                                <th>VIBRACIÓN / BIOMOLÉCULA</th>
                                {headers_samples}
                            </tr>
                        </thead>
                        <tbody>
                            {"".join(common_rows)}
                            {"".join(shared_rows)}
                            {"".join(unique_rows_compiled)}
                        </tbody>
                    </table>

                    <div class="footer">Generado automáticamente bajo estándar de publicación científica - Hershell-Raman v9.7</div>
                </div>
            </div>
        </body>
        </html>"""
        return {"html": html_report}
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"detail": str(e)})


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

def get_treatment_metadata(data: ChemoRequest):
    r_min, r_max = data.params.range[0], data.params.range[1]
    scale = data.params.scale.upper() if data.params.scale != "none" else "Ninguno"
    metadata = f"Tratamiento: Escalado {scale} | Rango: {r_min:.1f} - {r_max:.1f} cm⁻¹"
    return metadata



@app.post("/api/pca")
async def calculate_pca(data: ChemoRequest):
    try:
        if len(data.spectra) < 2: return {"error": "Se requieren al menos 2 espectros."}
        names = [clean_sample_name(s.name) for s in data.spectra]
        Y, _ = prepare_chemometric_matrix(data)
        
        n_comps = min(2, Y.shape[0])
        pca = PCA(n_components=n_comps)
        scores = pca.fit_transform(Y)
        evr = pca.explained_variance_ratio_ * 100
        
        scores_out = []
        for i, n in enumerate(names):
            pc1 = float(scores[i][0])
            pc2 = float(scores[i][1]) if n_comps > 1 else 0.0
            clean = clean_sample_name(data.spectra[i].name)
            scores_out.append({
                "name": clean, 
                "scientific_name": clean,  # Sin HTML
                "pc1": pc1, 
                "pc2": pc2
            })
            
        return {
            "type": "pca",
            "scores": scores_out,
            "explained_variance": [float(evr[0]), float(evr[1]) if n_comps > 1 else 0.0],
            "metadata": get_treatment_metadata(data)
        }
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return JSONResponse(status_code=400, content={"detail": f"Error matemático PCA: {str(e)}"})

@app.post("/api/hca")
async def calculate_hca(data: ChemoRequest):
    try:
        import pandas as pd
        import numpy as np
        from scipy.cluster import hierarchy
        from scipy.spatial.distance import pdist
        import plotly.graph_objects as go
        import json
        
        if len(data.spectra) < 2: return {"error": "Se requieren al menos 2 espectros."}
        
        # Generar matriz quimiométrica Y (n_spectra x n_wavenumbers)
        Y, x_ref = prepare_chemometric_matrix(data)
        
        df = pd.DataFrame(Y.T, columns=[clean_sample_name(s.name) for s in data.spectra])
        
        # 1. SALVAR TODAS LAS MUESTRAS: Convertir comas a puntos y forzar a float
        # select_dtypes() está PROHIBIDO: elimina columnas silenciosamente
        df_clean = df.replace(',', '.', regex=True)  # Arregla formato latino (1,5 -> 1.5)
        df_clean = df_clean.apply(pd.to_numeric, errors='coerce').fillna(0)  # Todo texto -> 0
        
        X = df_clean.T.values
        etiquetas = [str(col) for col in df_clean.columns]  # Garantiza las 6 etiquetas
        
        # 2. Matemática Ward
        dist_matrix = pdist(X, metric='euclidean')
        Z = hierarchy.linkage(dist_matrix, method='ward')

        # 3. Extraer coordenadas y COLORES del árbol
        # Definir un umbral de color (70% de la distancia máxima) para crear los clústeres visuales
        umbral_color = 0.7 * max(Z[:, 2]) if len(Z) > 0 else 0
        dendro_data = hierarchy.dendrogram(Z, labels=etiquetas, no_plot=True, color_threshold=umbral_color)

        # 4. Dibujar trazos manualmente CON COLORES DINÁMICOS traducidos a Hex (Plotly no acepta formato Matplotlib)
        color_map = {
            'C0': '#1f77b4', 'C1': '#ff7f0e', 'C2': '#2ca02c', 'C3': '#d62728',
            'C4': '#9467bd', 'C5': '#8c564b', 'C6': '#e377c2', 'C7': '#7f7f7f',
            'C8': '#bcbd22', 'C9': '#17becf', 'b': '#1f77b4', 'g': '#2ca02c',
            'r': '#d62728', 'c': '#17becf', 'm': '#e377c2', 'y': '#bcbd22',
            'k': '#2c3e50'
        }
        fig = go.Figure()
        for i, d, c in zip(dendro_data['icoord'], dendro_data['dcoord'], dendro_data['color_list']):
            # Traducir el color. Si es un formato no mapeado, usar gris por defecto
            plotly_color = color_map.get(c, '#555555')
            fig.add_trace(go.Scatter(
                x=i, y=d, mode='lines',
                line=dict(color=plotly_color, width=2),
                showlegend=False,
                hoverinfo='none'
            ))

        # 5. Forzar ubicación de las etiquetas para que aparezcan TODAS
        tick_vals = [5 + 10 * i for i in range(len(dendro_data['ivl']))]
        
        fig.update_layout(
            title=f"Dendrograma HCA (Método Ward)<br><sup>{get_treatment_metadata(data)}</sup>",
            xaxis=dict(
                tickvals=tick_vals,
                ticktext=dendro_data['ivl'],
                showgrid=False,
                tickangle=45
            ),
            yaxis=dict(title="Distancia Euclidiana (Ward)"),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(b=200, l=60, r=40, t=80),
            font=dict(family='Open Sans', size=12, color='#334155')
        )
        
        fig_json = json.loads(fig.to_json())
        
        return {
            "type": "hca_plotly",
            "figure": fig_json,
            "metadata": get_treatment_metadata(data)
        }
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return JSONResponse(status_code=400, content={"detail": f"Error matemático HCA: {str(e)}"})
@app.post("/api/correlation")
async def calculate_correlation(data: ChemoRequest):
    try:
        if len(data.spectra) < 2: return {"error": "Se requieren al menos 2 espectros."}
        orig_names = [s.name for s in data.spectra]
        # Nombres limpios sin HTML para etiquetas del heatmap
        names = [clean_sample_name(n) for n in orig_names]
        Y, _ = prepare_chemometric_matrix(data)
        
        corr_matrix = np.corrcoef(Y)
        return {
            "type": "correlation",
            "matrix": corr_matrix.tolist(),
            "labels": names,
            "metadata": get_treatment_metadata(data)
        }
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"detail": f"Error matemático Correlación: {str(e)}"})

class SpectrumItem(BaseModel):
    name: str = ""
    x: List[float]
    y: List[float]
    label: str = ""

class PreprocessPayload(BaseModel):
    spectra: List[SpectrumItem]

@app.post("/api/preprocessed_data")
async def get_preprocessed_data(payload: PreprocessPayload):
    try:
        espectros_payload = []
        for s in payload.spectra:
            espectros_payload.append({
                "wavenumbers": s.x,
                "absorbances": s.y,
                "label": s.label or s.name
            })
        if not espectros_payload:
            return {"error": "No data"}
        X_deriv, labels = aplicar_quimiometria(espectros_payload)
        x_ref = np.linspace(900.0, 4000.0, 1550)
        
        result = []
        for i in range(X_deriv.shape[0]):
            result.append({
                "name": labels[i],
                "x": x_ref.tolist(),
                "y": X_deriv[i].tolist()
            })
        return {"processed": result}
    except Exception as e:
        import traceback
        return JSONResponse(status_code=400, content={"detail": f"Error: {str(e)}", "trace": traceback.format_exc()})

class PLSDAPayload(BaseModel):
    spectra: List[SpectrumItem]
    n_components: int = 5
    algorithm: str = "pls_da"

@app.post("/api/pls_da")
async def calculate_pls_da(payload: PLSDAPayload):
    try:
        n_components = int(payload.n_components)
        algorithm = payload.algorithm
        
        if len(payload.spectra) < 3: return JSONResponse(status_code=400, content={"detail": "Se requieren al menos 3 espectros para entrenar."})
        
        espectros_payload = []
        names = []
        for s in payload.spectra:
            espectros_payload.append({
                "wavenumbers": s.x,
                "absorbances": s.y,
                "label": s.label
            })
            names.append(s.name)
            
        Y_features, labels_raw = aplicar_quimiometria(espectros_payload)
        x_ref = np.linspace(900.0, 4000.0, 1550)
        
        le = LabelBinarizer()
        Y_target = le.fit_transform(labels_raw)
        
        n_comps = max(1, min(n_components, Y_features.shape[0]-1))
        
        if algorithm.lower() in ["pls_da", "pls-da"]:
            pls = PLSRegression(n_components=n_comps)
            pls.fit(Y_features, Y_target)
            scores = pls.x_scores_
            plot_type = "Proyección PLS"
            
            if pls.coef_.ndim > 1:
                weights = np.mean(np.abs(pls.coef_), axis=1)
            else:
                weights = np.abs(pls.coef_).flatten()
                
        elif algorithm == 'Regresión Logística Penalizada (Elastic Net)':
            pipeline_elastic = Pipeline([
                ('scaler', StandardScaler()),
                ('logreg', LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, class_weight='balanced', max_iter=10000))
            ])
            Y_target_1d = np.argmax(Y_target, axis=1) if Y_target.ndim > 1 and Y_target.shape[1] > 1 else Y_target.flatten()
            encoder_estricto = LabelEncoder()
            Y_target_1d = encoder_estricto.fit_transform(Y_target_1d)
            
            pipeline_elastic.fit(Y_features, Y_target_1d)
            pca = PCA(n_components=2)
            scores = pca.fit_transform(Y_features)
            plot_type = "Proyección PCA"
            
            clf = pipeline_elastic.named_steps['logreg']
            if clf.coef_.ndim > 1:
                weights = np.mean(np.abs(clf.coef_), axis=0)
            else:
                weights = np.abs(clf.coef_).flatten()
                
        elif algorithm.strip().lower() == 'pca-lda':
            pipeline_lda = Pipeline([
                ('pca', PCA(n_components=n_comps)),
                ('lda', LinearDiscriminantAnalysis())
            ])
            Y_target_1d = np.argmax(Y_target, axis=1) if Y_target.ndim > 1 and Y_target.shape[1] > 1 else Y_target.flatten()
            encoder_estricto = LabelEncoder()
            Y_target_1d = encoder_estricto.fit_transform(Y_target_1d)
            
            pipeline_lda.fit(Y_features, Y_target_1d)
            
            pca_step = pipeline_lda.named_steps['pca']
            scores = pipeline_lda.transform(Y_features)
            plot_type = "Proyección PCA"
            
            weights = np.mean(np.abs(pca_step.components_), axis=0)
            
        else:
            return {"error": f"Algoritmo no soportado: {algorithm}"}
            
        # Normalización Z-Score de las Variables Latentes para evitar el solapamiento visual
        if scores.shape[0] > 1:
            scores = (scores - np.mean(scores, axis=0)) / (np.std(scores, axis=0) + 1e-8)
            
        scores_grouped = {}
        for i, grp in enumerate(labels_raw):
            if grp not in scores_grouped:
                scores_grouped[grp] = []
            scores_grouped[grp].append({
                "name": clean_sample_name(names[i]),
                "scientific_name": clean_sample_name(names[i]),
                "lv1": float(scores[i, 0]) if scores.shape[1] > 0 else 0.0,
                "lv2": float(scores[i, 1]) if scores.shape[1] > 1 else 0.0
            })
            
        return {
            "scores": scores_grouped,
            "weights": {"x": x_ref.tolist(), "y": weights.tolist()},
            "plot_type": plot_type
        }
    except Exception as e:
        import traceback
        return JSONResponse(
            status_code=400,
            content={"detail": f"Error interno PLS-DA: {str(e)}", "trace": traceback.format_exc()},
            headers={"Access-Control-Allow-Origin": "*"}
        )

class PredictPayload(BaseModel):
    train_spectra: List[SpectrumItem]
    test_spectra: List[SpectrumItem]
    n_components: int = 5
    algorithm: str = "pls_da"

@app.post("/api/predict")
async def predict_plsda(payload: PredictPayload):
    try:
        n_components = int(payload.n_components)
        import traceback
        if len(payload.train_spectra) < 3: return JSONResponse(status_code=400, content={"detail": "Se requieren al menos 3 espectros de entrenamiento."})
        if len(payload.test_spectra) < 1: return JSONResponse(status_code=400, content={"detail": "No hay espectros para predecir."})
        
        raw_train = []
        for s in payload.train_spectra:
            raw_train.append({"wavenumbers": s.x, "absorbances": s.y, "label": s.label})
            
        raw_test = []
        test_names_valid = []
        for s in payload.test_spectra:
            raw_test.append({"wavenumbers": s.x, "absorbances": s.y, "label": ""})
            test_names_valid.append(s.name)
            
        all_spectra = raw_train + raw_test
        Y_all, labels_all = aplicar_quimiometria(all_spectra)
        
        labels_raw = [s['label'] for s in raw_train]
        le = LabelBinarizer()
        Y_target = le.fit_transform(labels_raw)
        
        n_train = len(raw_train)
        Y_features_train = Y_all[:n_train]
        Y_features_test = Y_all[n_train:]
        
        n_comps = max(1, min(n_components, Y_features_train.shape[0]-1))
        
        if payload.algorithm.lower() in ["pls_da", "pls-da"]:
            try:
                pls = PLSRegression(n_components=n_comps)
                pls.fit(Y_features_train, Y_target)
                Y_pred = pls.predict(Y_features_test)
                
                if len(le.classes_) > 2:
                    pred_indices = np.argmax(Y_pred, axis=1)
                    predictions = le.classes_[pred_indices]
                elif len(le.classes_) == 2:
                    pred_indices = (Y_pred > 0.5).astype(int).flatten()
                    predictions = le.classes_[pred_indices]
                else:
                    predictions = [le.classes_[0]] * len(Y_pred)
            except Exception as e:
                return JSONResponse(status_code=400, content={"detail": f"Error interno de varianza (PLS-DA colapsó): {str(e)}"})
                
        elif payload.algorithm == 'Regresión Logística Penalizada (Elastic Net)':
            if len(le.classes_) < 2:
                predictions = [le.classes_[0]] * len(Y_features_test)
            else:
                pipeline_elastic = Pipeline([
                    ('scaler', StandardScaler()),
                    ('logreg', LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, class_weight='balanced', max_iter=10000))
                ])
                Y_target_1d = np.argmax(Y_target, axis=1) if Y_target.ndim > 1 and Y_target.shape[1] > 1 else Y_target.flatten()
                
                # Sanitización estricta de tipos para compatibilidad total con Pipelines
                encoder_estricto = LabelEncoder()
                Y_target_1d = encoder_estricto.fit_transform(Y_target_1d)
                
                try:
                    pipeline_elastic.fit(Y_features_train, Y_target_1d)
                    Y_pred = pipeline_elastic.predict(Y_features_test)
                    pred_indices = Y_pred.astype(int).flatten()
                    predictions = le.classes_[pred_indices]
                except Exception as e:
                    print(f"ElasticNet prediction fallback: {e}")
                    predictions = [le.classes_[0]] * len(Y_features_test)
            
        elif payload.algorithm.strip().lower() == 'pca-lda':
            if len(le.classes_) < 2:
                predictions = [le.classes_[0]] * len(Y_features_test)
            else:
                pipeline_lda = Pipeline([
                    ('pca', PCA(n_components=n_comps)),
                    ('lda', LinearDiscriminantAnalysis())
                ])
                Y_target_1d = np.argmax(Y_target, axis=1) if Y_target.ndim > 1 and Y_target.shape[1] > 1 else Y_target.flatten()
                
                # Sanitización estricta de tipos para compatibilidad total con Pipelines
                encoder_estricto = LabelEncoder()
                Y_target_1d = encoder_estricto.fit_transform(Y_target_1d)
                
                try:
                    pipeline_lda.fit(Y_features_train, Y_target_1d)
                    Y_pred = pipeline_lda.predict(Y_features_test)
                    pred_indices = Y_pred.astype(int).flatten()
                    predictions = le.classes_[pred_indices]
                except Exception as e:
                    print(f"PCA-LDA prediction fallback: {e}")
                    predictions = [le.classes_[0]] * len(Y_features_test)
            
        else:
            return JSONResponse(status_code=400, content={"detail": f"Algoritmo no soportado: {payload.algorithm}"})
            
        # Reconstruir la lista de predicciones considerando los archivos fallidos
        final_predictions = []
        pred_idx = 0
        for s in payload.test_spectra:
            if s.name in test_names_valid:
                final_predictions.append(predictions[pred_idx])
                pred_idx += 1
            else:
                final_predictions.append("Error_Lectura")
                
        res_data = final_predictions
        return {"predictions": res_data}
    except Exception as e:
        import traceback
        return {"error": str(e), "trace": traceback.format_exc()}

