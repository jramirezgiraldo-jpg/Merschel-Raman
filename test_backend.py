import os
import sys
import glob
import numpy as np

sys.path.append(r"C:\Users\Juan Felipe\.gemini\antigravity\scratch\Merschel-Raman\backend")
from main import procesar_mallas_y_aplicar_snv, parse_raw_spectroscopy_file

def load_folder(folder_path, label):
    files = glob.glob(os.path.join(folder_path, "*.csv")) + glob.glob(os.path.join(folder_path, "*.txt"))
    data = []
    for f in files:
        with open(f, 'rb') as file:
            try:
                w, a = parse_raw_spectroscopy_file(file.read(), os.path.basename(f))
                data.append({"wavenumbers": w.tolist(), "absorbances": a.tolist(), "label": label})
            except Exception as e:
                print(f"Error parsing {f}: {e}")
    return data

train_cryp = load_folder(r"C:\Users\Juan Felipe\Documents\1 Pasantia 2025 experimentos\Cryptosporidium FTIR microscopio Reims\Cryp_2", "Cryptosporidium")
train_gia = load_folder(r"C:\Users\Juan Felipe\Documents\1 Pasantia 2025 experimentos\Giardia FTIR microscopio Reims\Gia_1", "Giardia")
train_tox = load_folder(r"C:\Users\Juan Felipe\Documents\1 Pasantia 2025 experimentos\Toxoplasma FTIR microscopio Reims\Tox_1", "Toxoplasma")

all_train = train_cryp + train_gia + train_tox
print(f"Cargados {len(all_train)} espectros de entrenamiento.")

if len(all_train) > 0:
    try:
        X_snv, y_labels = procesar_mallas_y_aplicar_snv(all_train)
        print("PROCESAMIENTO SNV EXITOSO:")
        print(f"Dimensiones de X_snv: {X_snv.shape}")
        nan_count = np.isnan(X_snv).sum()
        print(f"NaNs en matriz: {nan_count}")
        if nan_count == 0:
            print("LA CORRECCION DE NANS FUE EXITOSA!")
    except Exception as e:
        print(f"ERROR DURANTE EL PROCESAMIENTO: {e}")
else:
    print("No se cargaron espectros.")
