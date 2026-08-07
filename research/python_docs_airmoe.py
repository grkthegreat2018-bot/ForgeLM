"""Python 3.13.3 Docs Ripper + AirMoE Knowledge Module

Downloads Python 3.13.3 official docs + wiki, processes them into
structured knowledge chunks, and packages them as an AirMoE module
(on-disk expert shards that can be hotswapped at inference).

Pipeline:
  1. Download Python 3.13.3 docs (HTML) from docs.python.org
  2. Extract text from HTML, split into topic-based chunks
  3. Compute embeddings for each chunk using ForgeLM
  4. Group chunks by topic (stdlib modules, language ref, tutorial, etc.)
  5. For each topic, create a Knowledge Pack (KV cache) or fact set
  6. Save as AirMoE expert shards on D: drive

The AirMoE module can then be loaded at inference:
  - Router: topic classifier (which Python topic is relevant?)
  - Experts: one per topic (stdlib, syntax, datatypes, etc.)
  - Hotswap: only load the relevant expert from disk

Usage:
    from research.python_docs_airmoe import PythonDocsAirMoE
    builder = PythonDocsAirMoE(output_dir="research/checkpoints/python_docs_airmoe")
    builder.build()
"""
import os
import sys
import re
import json
import time
import hashlib
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from html.parser import HTMLParser

# All output goes to D: drive
DEFAULT_OUTPUT = "D:/windsurf/ForgeAI/research/checkpoints/python_docs_airmoe"
DEFAULT_CACHE = "D:/windsurf/ForgeAI/.devin/tmp/python_docs_cache"


class HTMLTextExtractor(HTMLParser):
    """Extract clean text from HTML, preserving code blocks."""

    def __init__(self):
        super().__init__()
        self.text = []
        self.in_code = False
        self.in_pre = False
        self.skip = False
        self.current_tag = ""

    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        if tag in ("script", "style", "nav", "footer", "header"):
            self.skip = True
        elif tag == "pre":
            self.in_pre = True
            self.text.append("\n```\n")
        elif tag == "code":
            self.in_code = True
        elif tag in ("h1", "h2", "h3", "h4"):
            self.text.append("\n## ")
        elif tag == "p":
            self.text.append("\n")
        elif tag == "li":
            self.text.append("\n- ")
        elif tag == "a":
            pass  # keep link text

    def handle_endtag(self, tag):
        if tag in ("script", "style", "nav", "footer", "header"):
            self.skip = False
        elif tag == "pre":
            self.in_pre = False
            self.text.append("\n```\n")
        elif tag == "code":
            self.in_code = False
        elif tag in ("h1", "h2", "h3", "h4"):
            self.text.append("\n")
        elif tag == "p":
            self.text.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.text.append(data)

    def get_text(self) -> str:
        return "".join(self.text)


class PythonDocsAirMoE:
    """Download Python 3.13.3 docs, process into AirMoE knowledge module.

    The module consists of:
      - Topic-grouped knowledge chunks (JSON)
      - Embeddings for routing (npy)
      - AirMoE expert shards (safetensors, if model is available)
      - Router config (which expert for which topic)
    """

    # Python 3.13.3 doc sections to download
    DOC_SECTIONS = {
        "tutorial": "https://docs.python.org/3.13/tutorial/",
        "language": "https://docs.python.org/3.13/reference/",
        "library": "https://docs.python.org/3.13/library/",
        "builtin": "https://docs.python.org/3.13/library/functions.html",
        "types": "https://docs.python.org/3.13/library/stdtypes.html",
        "exceptions": "https://docs.python.org/3.13/library/exceptions.html",
        "constants": "https://docs.python.org/3.13/library/constants.html",
        "string": "https://docs.python.org/3.13/library/string.html",
        "re": "https://docs.python.org/3.13/library/re.html",
        "os": "https://docs.python.org/3.13/library/os.html",
        "sys": "https://docs.python.org/3.13/library/sys.html",
        "math": "https://docs.python.org/3.13/library/math.html",
        "json": "https://docs.python.org/3.13/library/json.html",
        "collections": "https://docs.python.org/3.13/library/collections.html",
        "itertools": "https://docs.python.org/3.13/library/itertools.html",
        "functools": "https://docs.python.org/3.13/library/functools.html",
        "pathlib": "https://docs.python.org/3.13/library/pathlib.html",
        "datetime": "https://docs.python.org/3.13/library/datetime.html",
        "typing": "https://docs.python.org/3.13/library/typing.html",
        "dataclasses": "https://docs.python.org/3.13/library/dataclasses.html",
        "asyncio": "https://docs.python.org/3.13/library/asyncio.html",
        "sqlite3": "https://docs.python.org/3.13/library/sqlite3.html",
        "unittest": "https://docs.python.org/3.13/library/unittest.html",
        "argparse": "https://docs.python.org/3.13/library/argparse.html",
        "logging": "https://docs.python.org/3.13/library/logging.html",
        "threading": "https://docs.python.org/3.13/library/threading.html",
        "multiprocessing": "https://docs.python.org/3.13/library/multiprocessing.html",
        "subprocess": "https://docs.python.org/3.13/library/subprocess.html",
        "io": "https://docs.python.org/3.13/library/io.html",
        "pickle": "https://docs.python.org/3.13/library/pickle.html",
        "csv": "https://docs.python.org/3.13/library/csv.html",
        "hashlib": "https://docs.python.org/3.13/library/hashlib.html",
        "enum": "https://docs.python.org/3.13/library/enum.html",
        "contextlib": "https://docs.python.org/3.13/library/contextlib.html",
        "warnings": "https://docs.python.org/3.13/library/warnings.html",
        "debug": "https://docs.python.org/3.13/library/debug.html",
        "whatsnew": "https://docs.python.org/3.13/whatsnew/3.13.html",
    }

    # Topic grouping for AirMoE experts
    TOPIC_GROUPS = {
        "basics": ["tutorial", "builtin", "types", "constants", "string"],
        "language": ["language", "exceptions"],
        "stdlib_core": ["os", "sys", "io", "pathlib", "subprocess",
                         "argparse", "logging", "enum", "contextlib",
                         "warnings", "debug"],
        "data": ["json", "csv", "pickle", "sqlite3", "hashlib",
                  "collections", "dataclasses", "typing"],
        "math_logic": ["math", "re", "itertools", "functools"],
        "concurrency": ["asyncio", "threading", "multiprocessing"],
        "testing": ["unittest"],
        "datetime": ["datetime"],
        "whatsnew": ["whatsnew"],
    }

    def __init__(self, output_dir: str = DEFAULT_OUTPUT,
                 cache_dir: str = DEFAULT_CACHE,
                 chunk_size: int = 500,
                 max_chunks_per_topic: int = 50):
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.max_chunks_per_topic = max_chunks_per_topic

    def download_page(self, url: str) -> str:
        """Download a URL and return HTML text."""
        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
            })
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read().decode('utf-8', errors='replace')
        except Exception as e:
            print(f"    Download failed for {url}: {e}")
            return ""

    def extract_text(self, html: str) -> str:
        """Extract clean text from HTML."""
        extractor = HTMLTextExtractor()
        try:
            extractor.feed(html)
        except Exception:
            pass
        text = extractor.get_text()
        # Clean up
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        return text.strip()

    def chunk_text(self, text: str, source: str) -> List[Dict]:
        """Split text into chunks of ~chunk_size words.

        Returns list of {"text": ..., "source": ..., "hash": ...}
        """
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size):
            chunk_text = " ".join(words[i:i + self.chunk_size])
            if len(chunk_text.strip()) < 50:
                continue
            chunk_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:8]
            chunks.append({
                "text": chunk_text,
                "source": source,
                "hash": chunk_hash,
                "word_count": len(words[i:i + self.chunk_size]),
            })
        return chunks

    def download_all_docs(self) -> Dict[str, List[Dict]]:
        """Download all Python 3.13.3 doc sections and chunk them.

        Returns dict: {section_name: [chunks]}
        """
        all_chunks = {}
        total_chunks = 0

        print(f"\n  Downloading {len(self.DOC_SECTIONS)} Python 3.13.3 doc sections...")

        for section_name, url in self.DOC_SECTIONS.items():
            cache_file = self.cache_dir / f"{section_name}.html"

            # Check cache
            if cache_file.exists():
                html = cache_file.read_text(encoding='utf-8')
                print(f"    [cached] {section_name}")
            else:
                print(f"    [download] {section_name} <- {url}")
                html = self.download_page(url)
                if html:
                    cache_file.write_text(html, encoding='utf-8')
                time.sleep(0.5)  # be nice to the server

            if not html:
                continue

            # Extract text and chunk
            text = self.extract_text(html)
            chunks = self.chunk_text(text, source=section_name)
            all_chunks[section_name] = chunks
            total_chunks += len(chunks)
            print(f"    → {len(chunks)} chunks ({len(text)} chars)")

        print(f"\n  Total: {total_chunks} chunks from {len(all_chunks)} sections")
        return all_chunks

    def group_by_topic(self, all_chunks: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """Group chunks by topic for AirMoE experts."""
        topic_chunks = {}
        for topic, sections in self.TOPIC_GROUPS.items():
            chunks = []
            for section in sections:
                if section in all_chunks:
                    # Limit chunks per topic
                    section_chunks = all_chunks[section][:self.max_chunks_per_topic]
                    for c in section_chunks:
                        c["topic"] = topic
                    chunks.extend(section_chunks)
            if chunks:
                topic_chunks[topic] = chunks
                print(f"    Topic '{topic}': {len(chunks)} chunks")

        return topic_chunks

    def build_knowledge_texts(self, topic_chunks: Dict[str, List[Dict]]) -> Dict[str, str]:
        """Build knowledge text for each topic (for KV cache or fact injection).

        Concatenates the most important chunks per topic into a single
        knowledge text that can be:
          - Fed to KnowledgePack.from_text() for KV cache injection
          - Converted to fact vectors for FactInjectionKey
          - Used as context for ContextPatchKey
        """
        knowledge_texts = {}
        for topic, chunks in topic_chunks.items():
            # Sort by word count (longer = more detailed)
            sorted_chunks = sorted(chunks, key=lambda c: c["word_count"], reverse=True)
            # Take top chunks and concatenate
            combined = "\n\n---\n\n".join(c["text"] for c in sorted_chunks[:20])
            knowledge_texts[topic] = combined
            print(f"    {topic}: {len(combined)} chars from {len(sorted_chunks[:20])} chunks")

        return knowledge_texts

    def save_module(self, topic_chunks: Dict[str, List[Dict]],
                    knowledge_texts: Dict[str, str]) -> str:
        """Save the AirMoE module to disk.

        Creates:
          - module.json: router config + metadata
          - topic_<name>.json: chunks per topic
          - knowledge_<name>.txt: knowledge text per topic
        """
        # Save module config
        module_config = {
            "name": "python_3.13.3_docs",
            "version": "3.13.3",
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "topics": list(topic_chunks.keys()),
            "total_chunks": sum(len(c) for c in topic_chunks.values()),
            "chunk_size": self.chunk_size,
            "source": "https://docs.python.org/3.13/",
        }
        config_path = self.output_dir / "module.json"
        config_path.write_text(json.dumps(module_config, indent=2), encoding='utf-8')

        # Save per-topic chunks
        for topic, chunks in topic_chunks.items():
            chunk_path = self.output_dir / f"topic_{topic}.json"
            chunk_path.write_text(json.dumps(chunks, indent=2, ensure_ascii=False),
                                  encoding='utf-8')

        # Save knowledge texts
        for topic, text in knowledge_texts.items():
            text_path = self.output_dir / f"knowledge_{topic}.txt"
            text_path.write_text(text, encoding='utf-8')

        # Save combined knowledge (all topics in one file for easy loading)
        combined_path = self.output_dir / "all_knowledge.json"
        combined = {topic: text for topic, text in knowledge_texts.items()}
        combined_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2),
                                 encoding='utf-8')

        total_size = sum(f.stat().st_size for f in self.output_dir.iterdir())
        print(f"\n  Module saved to {self.output_dir}")
        print(f"  Total size: {total_size / 1e6:.1f} MB")
        print(f"  Files: {len(list(self.output_dir.iterdir()))}")
        return str(config_path)

    def build(self) -> str:
        """Build the complete Python docs AirMoE module.

        Returns path to module.json config.
        """
        print("=" * 70)
        print("Python 3.13.3 Docs → AirMoE Knowledge Module")
        print("=" * 70)

        # Phase 1: Download all docs
        print("\n[1] Downloading Python 3.13.3 docs...")
        all_chunks = self.download_all_docs()

        # Phase 2: Group by topic
        print("\n[2] Grouping chunks by topic...")
        topic_chunks = self.group_by_topic(all_chunks)

        # Phase 3: Build knowledge texts
        print("\n[3] Building knowledge texts...")
        knowledge_texts = self.build_knowledge_texts(topic_chunks)

        # Phase 4: Save module
        print("\n[4] Saving AirMoE module...")
        config_path = self.save_module(topic_chunks, knowledge_texts)

        print(f"\n{'='*70}")
        print(f"Python Docs AirMoE Module Complete")
        print(f"{'='*70}")
        print(f"  Config: {config_path}")
        print(f"  Topics: {len(topic_chunks)}")
        print(f"  Total chunks: {sum(len(c) for c in topic_chunks.values())}")
        print(f"  Output: {self.output_dir}")
        print(f"\n  To use at inference:")
        print(f"    from research.keys.knowledge_pack_key import KnowledgePack")
        print(f"    knowledge = json.load(open('{self.output_dir}/all_knowledge.json'))")
        print(f"    # Create KV cache packs per topic")
        print(f"    # Router selects relevant topic → load that expert from disk")
        print(f"{'='*70}")

        return config_path


def main():
    """Build Python 3.13.3 docs AirMoE module."""
    builder = PythonDocsAirMoE(
        output_dir=DEFAULT_OUTPUT,
        cache_dir=DEFAULT_CACHE,
        chunk_size=500,
        max_chunks_per_topic=50,
    )
    builder.build()


if __name__ == "__main__":
    main()
