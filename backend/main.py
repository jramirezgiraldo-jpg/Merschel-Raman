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
def clean_sample_name(name: str):
    """
    Limpia el nombre del archivo eliminando extensiones y sufijos comunes.
    Utiliza os.path.splitext para mayor robustez estructural.
    """
    import os
    # Eliminar extensión principal
    base = os.path.basename(name)
    clean = os.path.splitext(base)[0]
    # Eliminar extensiones secundarias si existen (ej .txt.txt)
    while True:
        root, ext = os.path.splitext(clean)
        if ext.lower() in ['.csv', '.txt', '.asc', '.dat']:
            clean = root
        else:
            break
    # Eliminar guiones bajos o sufijos comunes
    clean = re.sub(r'(_|-)(raman|ftir|muestra|raw|proc|corr)\d*', '', clean, flags=re.IGNORECASE)
    # Reemplazar guiones bajos por espacios para un nombre limpio
    clean = clean.replace('_', ' ').strip()
    return clean

def format_scientific_name(name: str, use_latex: bool = False):
    """
    Aplica formato de cursiva taxonómica OBLIGATORIO para Plotly/HTML.
    """
    clean = clean_sample_name(name)
    if use_latex:
        parts = clean.split()
        if len(parts) >= 2:
            tex_name = r"\ ".join(parts)
            return f"$\\mathit{{{tex_name}}}$"
        return clean
    else:
        # Envoltorio HTML para renderizado en Plotly
        return f"<i>{clean}</i>"

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

class CharacterizeRequest(BaseModel):
    spectra: list[SpectrumInput]
    prominence: float = 0.05
    method: str = "Raman"

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
                "name": format_scientific_name(request.spectra[idx].name),
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
        diff_peaks_result[format_scientific_name(spec.name)] = unique_peaks
        
    return {"diff_peaks": diff_peaks_result}

@app.post("/api/characterize")
async def characterize_spectra(request: CharacterizeRequest):
    try:
        if not request.spectra:
            return {"error": "No hay espectros para caracterizar."}
            
        # 1. Alineación y Matrix Builder
        Y_matrix, x_ref = build_symmetric_matrix(request.spectra)
        names = [format_scientific_name(s.name) for s in request.spectra]
        
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
            if 1650 <= wn <= 1675: return "Estiramiento C=O: Amida I (Proteínas)"
            if 1540 <= wn <= 1560: return "Deformación N-H: Amida II (Proteínas)"
            if 1440 <= wn <= 1455: return "Deformación CH2: Flexión de cadenas (Lípidos)"
            if 1240 <= wn <= 1265: return "Estiramiento P=O asimétrico: Fosfodiéster (Ácidos Nucleicos)"
            if 1070 <= wn <= 1090: return "Estiramiento C-O: Enlaces Glicosídicos (Carbohidratos)"
            if 1003 <= wn <= 1005: return "Respiración del anillo: Fenilalanina (Proteínas)"
            if 930 <= wn <= 950:   return "Vibración C-C: Esqueleto proteico (α-hélice)"
            
            # REGIONES BIOQUÍMICAS (FALLBACK ESPECÍFICO)
            if 1500 <= wn <= 1800: return "Región de Proteínas (Fingerprint)"
            if 1200 <= wn < 1500: return "Región de Lípidos / Ácidos Nucleicos"
            if 900 <= wn < 1200: return "Región de Carbohidratos / Ácidos Nucleicos"
            
            return "Señal vibracional en zona Fingerprint"

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
        
        # Nombres formateados para HTML (cursiva) y Plotly
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
                if x - current_group[-1] <= 5.0: # Tolerancia ±5 cm-1
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
                match = next((p for p in spectra_peak_details[s_idx] if abs(p["x"] - g_wn) <= 5.0), None)
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
                    
                    <table>
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

def save_plot_to_base64():
    import io
    import base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_str

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
            formatted = format_scientific_name(n)
            scores_out.append({
                "name": n, 
                "scientific_name": formatted,
                "pc1": pc1, 
                "pc2": pc2
            })
            
        # Generar Imagen Matplotlib para Estándar Científico
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(9, 6))
        plot_labels = [format_scientific_name(n, use_latex=True) for n in names]
        sns.scatterplot(x=scores[:, 0], y=scores[:, 1] if n_comps > 1 else np.zeros_like(scores[:, 0]), 
                        hue=plot_labels, palette='viridis', s=100, alpha=0.8)
        plt.title(f'Análisis de Componentes Principales (PCA)\n{get_treatment_metadata(data)}')
        plt.xlabel(f'PC1 ({evr[0]:.1f}%)')
        plt.ylabel(f'PC2 ({evr[1]:.1f}%)' if n_comps > 1 else 'PC2')
        plt.grid(True, linestyle='--', alpha=0.6)
        img_b64 = save_plot_to_base64()

        return {
            "type": "pca",
            "scores": scores_out,
            "explained_variance": [float(evr[0]), float(evr[1]) if n_comps > 1 else 0.0],
            "plot_image": img_b64,
            "metadata": get_treatment_metadata(data)
        }
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"detail": f"Error matemático PCA: {str(e)}"})

@app.post("/api/hca")
async def calculate_hca(data: ChemoRequest):
    try:
        if len(data.spectra) < 2: return {"error": "Se requieren al menos 2 espectros."}
        names = [clean_sample_name(s.name) for s in data.spectra]
        
        Y, _ = prepare_chemometric_matrix(data)
        
        Z = linkage(Y, method=data.linkage_method, metric='euclidean')
        
        # Generar Imagen Matplotlib para Estándar Científico
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 7))
        plot_labels = [format_scientific_name(n, use_latex=True) for n in names]
        dendrogram(Z, labels=plot_labels, orientation='top', color_threshold=data.color_threshold)
        plt.title(f"Dendrograma de Agrupamiento Jerárquico (HCA)\n{get_treatment_metadata(data)}")
        plt.ylabel("Distancia Euclidiana")
        img_b64 = save_plot_to_base64()

        # Re-generar para JSON coords (Plotly)
        ddata = dendrogram(
            Z, 
            labels=names, 
            no_plot=True,
            truncate_mode=None, 
            color_threshold=data.color_threshold
        )
        
        # Formatear nombres de las hojas (eje X del dendrograma)
        scientific_ivl = [format_scientific_name(n) for n in ddata['ivl']]
        
        return {
            "type": "hca",
            "icoord": ddata['icoord'],
            "dcoord": ddata['dcoord'],
            "ivl": scientific_ivl,
            "plot_image": img_b64,
            "metadata": get_treatment_metadata(data)
        }
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"detail": f"Error matemático HCA: {str(e)}"})

@app.post("/api/correlation")
async def calculate_correlation(data: ChemoRequest):
    try:
        if len(data.spectra) < 2: return {"error": "Se requieren al menos 2 espectros."}
        orig_names = [s.name for s in data.spectra]
        names = [format_scientific_name(n) for n in orig_names]
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
                "scientific_name": format_scientific_name(names[i]),
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
