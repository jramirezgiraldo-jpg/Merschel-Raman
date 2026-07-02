import os
import shutil
import zipfile
from bs4 import BeautifulSoup
from googletrans import Translator
import re

translator = Translator()

def safe_translate(text):
    if not text or not text.strip():
        return text
    try:
        translated = translator.translate(text, src='es', dest='en').text
        return translated
    except Exception as e:
        print(f"Translation failed for: {text[:30]}... Error: {e}")
        return text

def translate_html(filepath, out_filepath):
    print(f"Translating {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')
    
    # Translate text nodes
    for text_node in soup.find_all(text=True):
        if text_node.parent.name not in ['style', 'script', 'head', 'title', 'meta', '[document]']:
            original_text = text_node.string
            if original_text and original_text.strip():
                translated = safe_translate(original_text.strip())
                new_text = original_text.replace(original_text.strip(), translated)
                text_node.replace_with(new_text)

    # Translate attributes like placeholder, alt
    for tag in soup.find_all(True):
        for attr in ['placeholder', 'alt', 'title']:
            if tag.has_attr(attr):
                val = tag[attr]
                if val and val.strip():
                    tag[attr] = safe_translate(val)

    with open(out_filepath, 'w', encoding='utf-8') as f:
        f.write(str(soup))

def translate_python(filepath, out_filepath):
    print(f"Translating {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#'):
            # It's a comment
            comment_text = line[line.find('#')+1:]
            translated = safe_translate(comment_text)
            new_lines.append(line[:line.find('#')+1] + translated + '\n')
        else:
            new_lines.append(line)
            
    with open(out_filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

def main():
    base_dir = r"C:\Users\Juan Felipe\.gemini\antigravity\scratch\Merschel-Raman"
    target_dir = r"C:\Users\Juan Felipe\.gemini\antigravity\scratch\Hershell-Raman-English"
    
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    
    # Copy all files first
    shutil.copytree(base_dir, target_dir, dirs_exist_ok=True, ignore=shutil.ignore_patterns('.git', '.github', '__pycache__', 'node_modules', '.venv', 'venv'))
    
    # Translate index.html
    html_path = os.path.join(target_dir, 'public', 'index.html')
    if os.path.exists(html_path):
        translate_html(html_path, html_path)
        
    # Translate main.py
    py_path = os.path.join(target_dir, 'backend', 'main.py')
    if os.path.exists(py_path):
        translate_python(py_path, py_path)
        
    # Translate README
    readme_path = os.path.join(target_dir, 'README.md')
    if os.path.exists(readme_path):
        translate_python(readme_path, readme_path) # Simple line by line is okay for markdown if it's mostly text, but might break. Let's skip README translation to be safe, or just do it. We will skip README.
        
    # Zip it up
    zip_path = r"C:\Users\Juan Felipe\.gemini\antigravity\scratch\Hershell-Raman-English.zip"
    shutil.make_archive(zip_path.replace('.zip', ''), 'zip', target_dir)
    print(f"Project successfully translated and zipped at: {zip_path}")

if __name__ == "__main__":
    main()
