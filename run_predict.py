import os
import glob
import requests

def get_files(folder_path):
    return glob.glob(os.path.join(folder_path, "*.csv")) + glob.glob(os.path.join(folder_path, "*.txt"))

train_folders = {
    "Cryptosporidium": r"C:\Users\Juan Felipe\Documents\1 Pasantia 2025 experimentos\Cryptosporidium FTIR microscopio Reims\Cryp_2",
    "Giardia": r"C:\Users\Juan Felipe\Documents\1 Pasantia 2025 experimentos\Giardia FTIR microscopio Reims\Gia_1",
    "Toxoplasma": r"C:\Users\Juan Felipe\Documents\1 Pasantia 2025 experimentos\Toxoplasma FTIR microscopio Reims\Tox_1"
}

test_folders = {
    "Cryp_1 (Esperado: Cryptosporidium)": r"C:\Users\Juan Felipe\Documents\1 Pasantia 2025 experimentos\Cryptosporidium FTIR microscopio Reims\Cryp_1",
    "Gia_4 (Esperado: Giardia)": r"C:\Users\Juan Felipe\Documents\1 Pasantia 2025 experimentos\Giardia FTIR microscopio Reims\Gia_4",
    "Tox_5 (Esperado: Toxoplasma)": r"C:\Users\Juan Felipe\Documents\1 Pasantia 2025 experimentos\Toxoplasma FTIR microscopio Reims\Tox_5"
}

files_payload = []
data_payload = []

# Prepare Training
for label, folder in train_folders.items():
    paths = get_files(folder)
    for p in paths:
        files_payload.append(('train_files', (os.path.basename(p), open(p, 'rb'), 'text/csv')))
        data_payload.append(('train_labels', label))

# Prepare Testing
test_order = []
for test_name, folder in test_folders.items():
    paths = get_files(folder)
    for p in paths:
        files_payload.append(('test_files', (os.path.basename(p), open(p, 'rb'), 'text/csv')))
        test_order.append((os.path.basename(p), test_name))

data_payload.append(('n_components', 2))

print(f"Enviando {len(data_payload)-1} espectros de entrenamiento y {len(test_order)} muestras ciegas...")

try:
    response = requests.post("http://127.0.0.1:8000/api/predict", files=files_payload, data=data_payload)
    if response.status_code == 200:
        res = response.json()
        preds = res.get("predictions", [])
        
        print("\n=== RESULTADOS DEL ANALISIS CIEGO ===")
        correct = 0
        for i, (filename, expected) in enumerate(test_order):
            pred_label = preds[i] if i < len(preds) else "ERROR"
            print(f"Muestra: {filename} | Predicción: {pred_label} | {expected}")
            
            if "Cryptosporidium" in expected and pred_label == "Cryptosporidium": correct += 1
            elif "Giardia" in expected and pred_label == "Giardia": correct += 1
            elif "Toxoplasma" in expected and pred_label == "Toxoplasma": correct += 1
            
        print(f"\nPRECISION TOTAL: {correct}/{len(test_order)} ({(correct/len(test_order))*100:.1f}%)")
        print("=====================================")
    else:
        print(f"Error HTTP {response.status_code}: {response.text}")
except Exception as e:
    print(f"Fallo en la peticion HTTP: {e}")
