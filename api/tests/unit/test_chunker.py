import pytest
from app.services.chunker import (
    get_splitter,
    compute_start_line,
    chunk_file,
    chunk_files,
    CHUNK_SIZE,
    LANGUAGE_MAP,
)
from app.models.schemas import ScrapedFile, Chunk
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ── get_splitter ──────────────────────────────────────────────────────────────

class TestGetSplitter:

    def test_returns_splitter_for_supported_language(self):
        for language in LANGUAGE_MAP.keys():
            splitter = get_splitter(language)
            assert isinstance(splitter, RecursiveCharacterTextSplitter)

    def test_returns_generic_splitter_for_unsupported_language(self):
        splitter = get_splitter("css")
        assert isinstance(splitter, RecursiveCharacterTextSplitter)

    def test_returns_generic_splitter_for_unknown_language(self):
        splitter = get_splitter("cobol")
        assert isinstance(splitter, RecursiveCharacterTextSplitter)

    def test_splitter_respects_chunk_size(self):
        splitter = get_splitter("python")
        assert splitter._chunk_size == CHUNK_SIZE


# ── compute_start_line ────────────────────────────────────────────────────────

class TestComputeStartLine:

    def test_chunk_at_start_of_file(self):
        content = "line1\nline2\nline3"
        assert compute_start_line(content, "line1") == 1

    def test_chunk_in_middle_of_file(self):
        content = "line1\nline2\nline3"
        assert compute_start_line(content, "line2") == 2

    def test_chunk_at_end_of_file(self):
        content = "line1\nline2\nline3"
        assert compute_start_line(content, "line3") == 3

    def test_chunk_not_found_returns_zero(self):
        content = "line1\nline2"
        assert compute_start_line(content, "nonexistent") == 0

    def test_multiline_chunk(self):
        content = "line1\nline2\nline3\nline4"
        assert compute_start_line(content, "line2\nline3") == 2

    def test_empty_content(self):
        assert compute_start_line("", "anything") == 0


# ── chunk_file ────────────────────────────────────────────────────────────────

class TestChunkFile:

    def test_returns_list_of_chunks(self, sample_scraped_file):
        result = chunk_file(sample_scraped_file)
        assert isinstance(result, list)
        assert all(isinstance(c, Chunk) for c in result)

    def test_chunks_inherit_path(self, sample_scraped_file):
        result = chunk_file(sample_scraped_file)
        assert all(c.path == sample_scraped_file.path for c in result)

    def test_chunks_inherit_language(self, sample_scraped_file):
        result = chunk_file(sample_scraped_file)
        assert all(c.language == sample_scraped_file.language for c in result)

    def test_chunks_have_valid_line_numbers(self, sample_scraped_file):
        result = chunk_file(sample_scraped_file)
        for chunk in result:
            assert chunk.start_line >= 1
            assert chunk.end_line >= chunk.start_line

    def test_chunks_have_non_empty_content(self, sample_scraped_file):
        result = chunk_file(sample_scraped_file)
        assert all(c.content.strip() for c in result)

    def test_small_file_produces_single_chunk(self):
        file = ScrapedFile(
            path="small.py",
            content="x = 1\n",
            language="python",
        )
        result = chunk_file(file)
        assert len(result) == 1

    def test_large_file_produces_multiple_chunks(self):
        content = "x = 1\n" * 500
        file = ScrapedFile(
            path="large.py",
            content=content,
            language="python",
        )
        result = chunk_file(file)
        assert len(result) > 1

    def test_chunk_size_respected(self):
        content = "x = 1\n" * 500
        file = ScrapedFile(
            path="large.py",
            content=content,
            language="python",
        )
        result = chunk_file(file)
        for chunk in result:
            assert len(chunk.content) <= CHUNK_SIZE * 1.2

    @pytest.mark.parametrize("language", list(LANGUAGE_MAP.keys()))
    def test_all_supported_languages(self, language):
        file = ScrapedFile(
            path=f"main.{language}",
            content="some code content\n" * 10,
            language=language,
        )
        result = chunk_file(file)
        assert len(result) > 0
        assert all(isinstance(c, Chunk) for c in result)

    def test_unsupported_language_falls_back_to_generic_splitter(self):
        file = ScrapedFile(
            path="styles.css",
            content="body { margin: 0; }\n" * 10,
            language="css",
        )
        result = chunk_file(file)
        assert len(result) > 0

    def test_markdown_file_is_chunked(self):
        content = "# Title\n\n" + "Some content.\n" * 50
        file = ScrapedFile(
            path="README.md",
            content=content,
            language="markdown",
        )
        result = chunk_file(file)
        assert len(result) > 0


# ── chunk_files ───────────────────────────────────────────────────────────────

class TestChunkFiles:

    def test_empty_list_returns_empty(self):
        assert chunk_files([]) == []

    def test_single_file_returns_chunks(self, sample_scraped_file):
        result = chunk_files([sample_scraped_file])
        assert len(result) > 0
        assert all(isinstance(c, Chunk) for c in result)

    def test_multiple_files_chunks_are_combined(self):
        files = [
            ScrapedFile(path="a.py", content="x = 1\n", language="python"),
            ScrapedFile(path="b.py", content="y = 2\n", language="python"),
        ]
        result = chunk_files(files)
        paths = {c.path for c in result}
        assert "a.py" in paths
        assert "b.py" in paths

    def test_chunks_from_different_files_dont_mix(self):
        files = [
            ScrapedFile(path="a.py", content="x = 1\n", language="python"),
            ScrapedFile(path="b.ts", content="const y = 2;\n", language="typescript"),
        ]
        result = chunk_files(files)
        for chunk in result:
            if chunk.path == "a.py":
                assert chunk.language == "python"
            elif chunk.path == "b.ts":
                assert chunk.language == "typescript"

    def test_total_chunks_equals_sum_of_individual_files(self):
        files = [
            ScrapedFile(path="a.py", content="x = 1\n" * 200, language="python"),
            ScrapedFile(path="b.py", content="y = 2\n" * 200, language="python"),
        ]
        individual = sum(len(chunk_file(f)) for f in files)
        combined = len(chunk_files(files))
        assert combined == individual
