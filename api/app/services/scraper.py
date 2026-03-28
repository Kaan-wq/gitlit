from git import Repo
from app.models.schemas import ScrapedFile
import tempfile
import os

IGNORE_DIRS = {'.git', 'node_modules', '__pycache__', '.venv', 'dist', 'build', '.idea', '.vscode'}
IGNORE_FILES = {'.DS_Store', 'poetry.lock', 'package-lock.json', 'yarn.lock'}

EXTENSION_TO_LANGUAGE = {
    '.py': 'python',
    '.ts': 'typescript',
    '.tsx': 'typescript',
    '.js': 'javascript',
    '.jsx': 'javascript',
    '.md': 'markdown',
    '.java': 'java',
    '.go': 'go',
    '.rs': 'rust',
    '.cpp': 'cpp',
    '.c': 'c',
    '.html': 'html',
    '.css': 'css',
}

def scrape_repository(repo_url: str) -> list[ScrapedFile]:
    files = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        Repo.clone_from(repo_url, tmp_dir, depth=1)
        for root, dirs, filenames in os.walk(tmp_dir):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            for filename in filenames:
                if filename in IGNORE_FILES:
                    continue
                ext = os.path.splitext(filename)[1].lower()
                language = EXTENSION_TO_LANGUAGE.get(ext)
                if language is None:
                    continue
                abs_path = os.path.join(root, filename)
                rel_path = os.path.relpath(abs_path, tmp_dir)
                with open(abs_path, 'r', encoding='utf-8', errors='ignore') as fh:
                    content = fh.read()
                if not content.strip():
                    continue
                files.append(ScrapedFile(path=rel_path, content=content, language=language))
    return files
