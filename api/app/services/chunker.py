from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from app.models.schemas import Chunk, ScrapedFile

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

LANGUAGE_MAP = {
    "python": Language.PYTHON,
    "typescript": Language.TS,
    "javascript": Language.JS,
    "markdown": Language.MARKDOWN,
    "java": Language.JAVA,
    "go": Language.GO,
    "rust": Language.RUST,
    "cpp": Language.CPP,
    "c": Language.C,
    "html": Language.HTML,
}

def get_splitter(language: str) -> RecursiveCharacterTextSplitter:
    lang = LANGUAGE_MAP.get(language)
    if lang:
        return RecursiveCharacterTextSplitter.from_language(
            language=lang,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

def compute_start_line(content: str, chunk: str) -> int:
    idx = content.find(chunk)
    if idx == -1:
        return 0
    return content[:idx].count("\n") + 1

def chunk_file(file: ScrapedFile) -> list[Chunk]:
    content = file.content
    language = file.language
    path = file.path

    splitter = get_splitter(language)
    chunks = splitter.split_text(content)

    result = []
    for chunk in chunks:
        start_line = compute_start_line(content, chunk)
        end_line = start_line + chunk.count("\n")
        result.append(Chunk(
            path=path,
            content=chunk,
            language=language,
            start_line=start_line,
            end_line=end_line,
        ))
    return result

def chunk_files(files: list[ScrapedFile]) -> list[Chunk]:
    all_chunks = []
    for file in files:
        all_chunks.extend(chunk_file(file))
    return all_chunks
