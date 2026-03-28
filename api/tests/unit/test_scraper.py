import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock
from app.services.scraper import scrape_repository, IGNORE_DIRS, IGNORE_FILES
from app.models.schemas import ScrapedFile


def make_temp_repo(files: dict[str, str]) -> tempfile.TemporaryDirectory:
    """Helper — creates a temp directory with the given {relative_path: content} files."""
    tmp = tempfile.TemporaryDirectory()
    for rel_path, content in files.items():
        abs_path = os.path.join(tmp.name, rel_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)
    return tmp


def run_scraper_on_dir(tmp_dir: str) -> list[ScrapedFile]:
    """Helper — runs scrape_repository but bypasses the git clone."""
    with patch("app.services.scraper.Repo.clone_from"):
        with patch("tempfile.TemporaryDirectory") as mock_tmp:
            mock_tmp.return_value.__enter__ = lambda s: tmp_dir
            mock_tmp.return_value.__exit__ = MagicMock(return_value=False)
            return scrape_repository("https://github.com/fake/repo")


# ── Supported extensions ──────────────────────────────────────────────────────

class TestSupportedExtensions:

    @pytest.mark.parametrize("filename,expected_language", [
        ("main.py", "python"),
        ("index.ts", "typescript"),
        ("component.tsx", "typescript"),
        ("app.js", "javascript"),
        ("component.jsx", "javascript"),
        ("README.md", "markdown"),
        ("Main.java", "java"),
        ("main.go", "go"),
        ("main.rs", "rust"),
        ("main.cpp", "cpp"),
        ("main.c", "c"),
        ("index.html", "html"),
        ("styles.css", "css"),
    ])
    def test_supported_extension_is_scraped(self, filename, expected_language):
        with make_temp_repo({filename: "some content"}) as tmp_dir:
            results = run_scraper_on_dir(tmp_dir)
            assert len(results) == 1
            assert results[0].language == expected_language
            assert results[0].path == filename

    @pytest.mark.parametrize("filename", [
        "data.json",
        "image.png",
        "archive.zip",
        "binary.exe",
        "data.csv",
        "config.yaml",
        "config.toml",
        ".env",
    ])
    def test_unsupported_extension_is_skipped(self, filename):
        with make_temp_repo({filename: "some content"}) as tmp_dir:
            results = run_scraper_on_dir(tmp_dir)
            assert len(results) == 0


# ── Ignored files ─────────────────────────────────────────────────────────────

class TestIgnoredFiles:

    @pytest.mark.parametrize("filename", list(IGNORE_FILES))
    def test_ignored_file_is_skipped(self, filename):
        with make_temp_repo({filename: "some content"}) as tmp_dir:
            results = run_scraper_on_dir(tmp_dir)
            assert len(results) == 0

    def test_non_ignored_file_is_not_skipped(self):
        with make_temp_repo({"main.py": "print('hello')"}) as tmp_dir:
            results = run_scraper_on_dir(tmp_dir)
            assert len(results) == 1


# ── Ignored directories ───────────────────────────────────────────────────────

class TestIgnoredDirectories:

    @pytest.mark.parametrize("ignored_dir", list(IGNORE_DIRS))
    def test_ignored_directory_is_not_traversed(self, ignored_dir):
        with make_temp_repo({
            f"{ignored_dir}/main.py": "print('should be ignored')",
            "main.py": "print('should be scraped')",
        }) as tmp_dir:
            results = run_scraper_on_dir(tmp_dir)
            assert len(results) == 1
            assert results[0].path == "main.py"

    def test_nested_valid_files_are_scraped(self):
        with make_temp_repo({
            "src/utils/helper.py": "def helper(): pass",
            "src/main.py": "def main(): pass",
        }) as tmp_dir:
            results = run_scraper_on_dir(tmp_dir)
            assert len(results) == 2


# ── Empty files ───────────────────────────────────────────────────────────────

class TestEmptyFiles:

    def test_empty_file_is_skipped(self):
        with make_temp_repo({"main.py": ""}) as tmp_dir:
            results = run_scraper_on_dir(tmp_dir)
            assert len(results) == 0

    def test_whitespace_only_file_is_skipped(self):
        with make_temp_repo({"main.py": "   \n\n\t  "}) as tmp_dir:
            results = run_scraper_on_dir(tmp_dir)
            assert len(results) == 0

    def test_non_empty_file_is_not_skipped(self):
        with make_temp_repo({"main.py": "x = 1"}) as tmp_dir:
            results = run_scraper_on_dir(tmp_dir)
            assert len(results) == 1


# ── Return type and schema ────────────────────────────────────────────────────

class TestReturnSchema:

    def test_returns_list_of_scraped_files(self):
        with make_temp_repo({"main.py": "x = 1"}) as tmp_dir:
            results = run_scraper_on_dir(tmp_dir)
            assert isinstance(results, list)
            assert all(isinstance(f, ScrapedFile) for f in results)

    def test_scraped_file_has_correct_fields(self):
        with make_temp_repo({"main.py": "x = 1"}) as tmp_dir:
            results = run_scraper_on_dir(tmp_dir)
            f = results[0]
            assert f.path == "main.py"
            assert f.content == "x = 1"
            assert f.language == "python"

    def test_relative_path_not_absolute(self):
        with make_temp_repo({"src/main.py": "x = 1"}) as tmp_dir:
            results = run_scraper_on_dir(tmp_dir)
            assert not os.path.isabs(results[0].path)


# ── Invalid repo URL ──────────────────────────────────────────────────────────

class TestInvalidRepoUrl:

    def test_invalid_url_raises_exception(self):
        with pytest.raises(Exception):
            scrape_repository("https://github.com/fake/nonexistent-repo-xyz-123")

    def test_malformed_url_raises_exception(self):
        with pytest.raises(Exception):
            scrape_repository("not-a-url")
