import subprocess

try:
    subprocess.run(["git", "add", "public/logo.png"], check=True)
    subprocess.run(["git", "add", "public/index.html"], check=True)
    subprocess.run(["git", "commit", "-m", "UI FIX: Rediseño del modal a 2 columnas (Split Layout) y forzado de subida del archivo binario logo.png a GitHub."], check=True)
    subprocess.run(["git", "push", "origin", "HEAD"], check=True)
    print("Git operations successful.")
except subprocess.CalledProcessError as e:
    print(f"Error during git operations: {e}")
