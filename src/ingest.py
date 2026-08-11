"""
ingest.py — Document ingestion pipeline for IT Audit LLM
Parses audit documents, chunks them, embeds, and stores in ChromaDB.

Usage:
    python src/ingest.py --docs /path/to/audit/docs/
    python src/ingest.py --docs data/synthetic/ --reset
"""

import os
import json
import hashlib
import argparse
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.progress import track

console = Console()

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "audit_evidence"
EMBED_MODEL = "all-MiniLM-L6-v2"   # 80MB, runs on CPU, good quality
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


class AuditDocumentIngester:
    def __init__(self, chroma_path: str = CHROMA_PATH, reset: bool = False):
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.client = chromadb.PersistentClient(path=chroma_path)

        if reset:
            try:
                self.client.delete_collection(COLLECTION_NAME)
                console.print("[yellow]Existing collection deleted.[/yellow]")
            except Exception:
                pass

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        console.print(f"[green]ChromaDB ready at {chroma_path}[/green]")
        console.print(f"[green]Collection '{COLLECTION_NAME}' — {self.collection.count()} existing docs[/green]")

    def ingest_directory(self, docs_path: str) -> int:
        """Ingest all supported files from a directory."""
        docs_dir = Path(docs_path)
        files = list(docs_dir.rglob("*.pdf")) + \
                list(docs_dir.rglob("*.docx")) + \
                list(docs_dir.rglob("*.xlsx")) + \
                list(docs_dir.rglob("*.txt")) + \
                list(docs_dir.rglob("*.jsonl"))

        console.print(f"[blue]Found {len(files)} files to process[/blue]")
        total = 0
        for f in track(files, description="Ingesting documents..."):
            count = self.ingest_file(str(f))
            total += count
        console.print(f"[green]Done. {total} chunks stored in ChromaDB.[/green]")
        return total

    def ingest_file(self, file_path: str) -> int:
        """Parse a single file and store chunks."""
        path = Path(file_path)
        ext = path.suffix.lower()

        try:
            if ext == ".jsonl":
                chunks = self._parse_jsonl(file_path)
            elif ext in (".txt", ".md"):
                chunks = self._parse_text(file_path)
            elif ext == ".pdf":
                chunks = self._parse_pdf(file_path)
            elif ext == ".docx":
                chunks = self._parse_docx(file_path)
            elif ext == ".xlsx":
                chunks = self._parse_xlsx(file_path)
            else:
                return 0

            self._store_chunks(chunks, source=path.name)
            return len(chunks)

        except Exception as e:
            console.print(f"[red]Error processing {file_path}: {e}[/red]")
            return 0

    def _parse_jsonl(self, file_path: str) -> List[Dict]:
        """Parse audit Q&A JSONL training file into retrievable chunks."""
        chunks = []
        with open(file_path) as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    text = f"CONTROL: {record.get('control_id', '')}\n" \
                           f"DOMAIN: {record.get('domain', '')}\n" \
                           f"QUESTION: {record.get('question', '')}\n" \
                           f"FINDING: {record.get('answer', '')}"
                    chunks.append({
                        "text": text,
                        "control_id": record.get("control_id", ""),
                        "domain": record.get("domain", ""),
                        "source_type": "qa_pair",
                    })
                except json.JSONDecodeError:
                    continue
        return chunks

    def _parse_text(self, file_path: str) -> List[Dict]:
        """Parse plain text file into overlapping chunks."""
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return self._chunk_text(text, {"source_type": "text_doc"})

    def _parse_pdf(self, file_path: str) -> List[Dict]:
        """Parse PDF using PyMuPDF."""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(file_path)
            full_text = "\n".join(page.get_text() for page in doc)
            return self._chunk_text(full_text, {"source_type": "pdf"})
        except ImportError:
            console.print("[yellow]PyMuPDF not installed. pip install pymupdf[/yellow]")
            return []

    def _parse_docx(self, file_path: str) -> List[Dict]:
        """Parse Word document."""
        try:
            from docx import Document
            doc = Document(file_path)
            full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            return self._chunk_text(full_text, {"source_type": "docx"})
        except ImportError:
            console.print("[yellow]python-docx not installed. pip install python-docx[/yellow]")
            return []

    def _parse_xlsx(self, file_path: str) -> List[Dict]:
        """Parse Excel file — each sheet becomes a text block."""
        try:
            import pandas as pd
            chunks = []
            xl = pd.ExcelFile(file_path)
            for sheet in xl.sheet_names:
                df = xl.parse(sheet)
                text = f"Sheet: {sheet}\n{df.to_string(index=False)}"
                chunks.extend(self._chunk_text(text, {"source_type": "xlsx", "sheet": sheet}))
            return chunks
        except ImportError:
            console.print("[yellow]pandas/openpyxl not installed.[/yellow]")
            return []

    def _chunk_text(self, text: str, metadata: Dict) -> List[Dict]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk_words = words[i:i + CHUNK_SIZE]
            chunk_text = " ".join(chunk_words)
            if len(chunk_text.strip()) < 50:
                continue
            chunks.append({"text": chunk_text, **metadata})
        return chunks

    def _store_chunks(self, chunks: List[Dict], source: str):
        """Embed and store chunks in ChromaDB."""
        if not chunks:
            return

        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.encode(texts, show_progress_bar=False).tolist()

        ids, docs, metas, embeds = [], [], [], []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            doc_id = hashlib.md5(f"{source}_{i}_{chunk['text'][:50]}".encode()).hexdigest()
            ids.append(doc_id)
            docs.append(chunk["text"])
            metas.append({k: v for k, v in chunk.items() if k != "text"} | {"source": source})
            embeds.append(emb)

        self.collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeds)

    def retrieve(self, query: str, n_results: int = 5, domain: str = None) -> List[Dict]:
        """Retrieve relevant chunks for an audit query."""
        query_embedding = self.embedder.encode([query]).tolist()
        where = {"domain": domain} if domain else None

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where,
        )

        retrieved = []
        for i in range(len(results["documents"][0])):
            retrieved.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return retrieved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest audit documents into ChromaDB")
    parser.add_argument("--docs", required=True, help="Path to documents directory")
    parser.add_argument("--reset", action="store_true", help="Reset existing collection")
    args = parser.parse_args()

    ingester = AuditDocumentIngester(reset=args.reset)
    ingester.ingest_directory(args.docs)
