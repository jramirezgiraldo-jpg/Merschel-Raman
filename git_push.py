import subprocess

try:
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-m", "AUDIT & UI FIX: Auditoría superada sin daños. Modal de bienvenida inyectado con logo y manifiesto íntegro de privacidad e Ingeniería de Prompts."], check=True)
    subprocess.run(["git", "push", "origin", "HEAD"], check=True)
    print("Git operations successful.")
except subprocess.CalledProcessError as e:
    print(f"Error during git operations: {e}")
