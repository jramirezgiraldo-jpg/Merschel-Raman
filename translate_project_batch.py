import os
import shutil
import zipfile
from bs4 import BeautifulSoup
from googletrans import Translator
import time

translator = Translator()

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def translate_html(filepath, out_filepath):
    print(f"Translating {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')
    
    nodes_to_translate = []
    for text_node in soup.find_all(string=True):
        if text_node.parent.name not in ['style', 'script', 'head', 'title', 'meta', '[document]']:
            if text_node.string and text_node.string.strip():
                nodes_to_translate.append(text_node)

    # Batch translate text nodes
    batch_size = 50
    for batch in chunk_list(nodes_to_translate, batch_size):
        texts = [node.string.strip() for node in batch]
        try:
            translated_objs = translator.translate(texts, src='es', dest='en')
            if not isinstance(translated_objs, list):
                translated_objs = [translated_objs]
            for node, t_obj in zip(batch, translated_objs):
                new_text = node.string.replace(node.string.strip(), t_obj.text)
                node.replace_with(new_text)
        except Exception as e:
            print(f"Batch translation failed: {e}")
            time.sleep(1)

    with open(out_filepath, 'w', encoding='utf-8') as f:
        f.write(str(soup))

def translate_python(filepath, out_filepath):
    print(f"Translating {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    comments = []
    comment_indices = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('#'):
            comments.append(line[line.find('#')+1:])
            comment_indices.append(i)
            
    # Batch translate
    batch_size = 50
    for comments_batch, indices_batch in zip(chunk_list(comments, batch_size), chunk_list(comment_indices, batch_size)):
        try:
            translated_objs = translator.translate(comments_batch, src='es', dest='en')
            if not isinstance(translated_objs, list):
                translated_objs = [translated_objs]
            for idx, t_obj in zip(indices_batch, translated_objs):
                orig_line = lines[idx]
                lines[idx] = orig_line[:orig_line.find('#')+1] + t_obj.text + '\n'
        except Exception as e:
            print(f"Batch translation failed for python: {e}")
            time.sleep(1)
            
    with open(out_filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)

def main():
    base_dir = r"C:\Users\Juan Felipe\.gemini\antigravity\scratch\Merschel-Raman"
    target_dir = r"C:\Users\Juan Felipe\.gemini\antigravity\scratch\Hershell-Raman-English"
    
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    
    shutil.copytree(base_dir, target_dir, dirs_exist_ok=True, ignore=shutil.ignore_patterns('.git', '.github', '__pycache__', 'node_modules', '.venv', 'venv'))
    
    html_path = os.path.join(target_dir, 'public', 'index.html')
    if os.path.exists(html_path):
        translate_html(html_path, html_path)
        
    py_path = os.path.join(target_dir, 'backend', 'main.py')
    if os.path.exists(py_path):
        translate_python(py_path, py_path)
        
    zip_path = r"C:\Users\Juan Felipe\.gemini\antigravity\scratch\Hershell-Raman-English.zip"
    shutil.make_archive(zip_path.replace('.zip', ''), 'zip', target_dir)
    print(f"Project successfully translated and zipped at: {zip_path}")

if __name__ == "__main__":
    main()
