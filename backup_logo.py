import shutil
import os

source_dir = r"C:\Users\Juan Felipe\.gemini\antigravity\scratch\Merschel-Raman"
backup_dir = r"C:\Users\Juan Felipe\.gemini\antigravity\scratch\Hershell-Raman_Backup_Logo"

if not os.path.exists(backup_dir):
    shutil.copytree(source_dir, backup_dir, dirs_exist_ok=True)
    print(f"Backup creado exitosamente en {backup_dir}")
else:
    print("El backup ya existe.")
