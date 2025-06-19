"""
utils.py – utilidades para preprocesamiento de texto y generación de embeddings

Este módulo facilita las tareas recurrentes en el proyecto de detección de fake
news:

1. Limpieza de texto y tokenización.
2. Construcción de vocabulario (diccionario único de palabras).
3. Descarga de embeddings vía la API de OpenAI (text-embedding-ada-002).
4. Almacenamiento y carga de diccionarios de embeddings en formato JSON.

Ejemplo de uso rápido desde línea de comandos:

```bash
# Generar/actualizar un archivo embeddings.json a partir de True.csv y Fake.csv
python utils.py --input data/True.csv data/Fake.csv --output data/embeddings.json
```

Requisitos:
    pip install pandas tqdm nltk openai
    # Tras instalar NLTK por primera vez:
    python -m nltk.downloader stopwords punkt

La clave de API se lee desde la variable de entorno `OPENAI_API_KEY` o puede
pasarse con el flag `--api-key`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

import openai
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm.auto import tqdm

# ────────────────────────────────────────────────────────────────────────────────
# Text pre‑processing helpers
# ────────────────────────────────────────────────────────────────────────────────

_STOPWORDS = set(stopwords.words("english"))
_TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")  # only letters + apostrophes


def clean_text(text: str) -> str:
    """Minimiza puntuación, links, espacios extra y pasa a minúsculas."""
    text = re.sub(r"https?://\S+", "", text)  # remove URLs
    text = re.sub(r"[^\w\s']", " ", text)  # replace punctuation with space
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def tokenize(text: str) -> List[str]:
    """Tokeniza usando NLTK y descarta stopwords y tokens de una sola letra."""
    tokens = word_tokenize(text)
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def build_vocab(text_iter: Iterable[str], min_freq: int = 2) -> List[str]:
    """Crea vocabulario filtrando por frecuencia mínima."""
    counter: Counter[str] = Counter()
    for text in text_iter:
        counter.update(tokenize(clean_text(text)))
    return [w for w, c in counter.items() if c >= min_freq]

# ────────────────────────────────────────────────────────────────────────────────
# Embedding helpers
# ────────────────────────────────────────────────────────────────────────────────

def generate_embeddings(words: List[str], model: str = "text-embedding-ada-002") -> Dict[str, List[float]]:
    """Genera un embedding por palabra usando la API de OpenAI.

    Retorna un diccionario {palabra: vector}.
    """
    embeddings: Dict[str, List[float]] = {}
    total_tokens = 0

    for word in tqdm(words, desc="Generando embeddings", total=len(words)):
        response = openai.Embeddings.create(model=model, input=word)
        embeddings[word] = response["data"][0]["embedding"]
        total_tokens += response["usage"]["total_tokens"]

    tqdm.write(f"Tokens totales consumidos: {total_tokens:,}")
    return embeddings


# ────────────────────────────────────────────────────────────────────────────────
# JSON helpers
# ────────────────────────────────────────────────────────────────────────────────

def save_json(data: Dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def load_json(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ────────────────────────────────────────────────────────────────────────────────
# CLI entry‑point
# ────────────────────────────────────────────────────────────────────────────────

def _load_datasets(paths: List[str]) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in paths]
    return pd.concat(frames, ignore_index=True)


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Genera/actualiza embeddings JSON")
    parser.add_argument("--input", nargs="+", required=True, help="Ruta(s) CSV de datos")
    parser.add_argument("--output", required=True, help="Archivo JSON destino para embeddings")
    parser.add_argument("--min-freq", type=int, default=2, help="Frecuencia mínima de palabra para incluirla")
    parser.add_argument("--api-key", default=None, help="Clave de OpenAI; si no se pasa, se usa OPENAI_API_KEY del entorno")
    args = parser.parse_args(argv)

    openai.api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        sys.exit("[ERROR] Missing OpenAI API key (flag --api-key o variable OPENAI_API_KEY)")

    out_path = Path(args.output)
    if out_path.exists():
        print(f"[INFO] {out_path} ya existe. Cargando...")
        embeddings = load_json(out_path)
    else:
        print("[INFO] Generando nuevo archivo de embeddings…")
        df = _load_datasets(args.input)
        vocab = build_vocab(df["text"].fillna(""), min_freq=args.min_freq)
        embeddings = generate_embeddings(vocab)
        save_json(embeddings, out_path)
        print(f"[OK] Embeddings guardados en {out_path} ({len(embeddings)} palabras)")


if __name__ == "__main__":
    main()
