
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import unicodedata
import warnings
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import pdfplumber

HANZI_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
OXFORD_LEVEL_RE = re.compile(r"^(A1|A2|B1|B2|C1)$", re.I)
HSK_LEVEL_RE = re.compile(
    r"(?:NEW\s+HSK\s+VOCABULARY\s*)?"
    r"LEVEL\s*([1-9](?:\s*-\s*[1-9])?)",
    re.I,
)
HSK_ENTRY_RE = re.compile(
    r"^(?P<number>\d+)\s+"
    r"(?P<word>\S+)\s+"
    r"(?P<pinyin>\S+)"
    r"(?:\s+(?P<rest>.*))?$"
)

HSK_EXPECTED = {
    "HSK 1": 300,
    "HSK 2": 200,
    "HSK 3": 500,
    "HSK 4": 1000,
    "HSK 5": 1600,
    "HSK 6": 1800,
    "HSK 7-9": 5600,
}

ZH_POS = {
    "能愿": "modal",
    "名": "noun",
    "动": "verb",
    "形": "adjective",
    "副": "adverb",
    "代": "pronoun",
    "介": "preposition",
    "连": "conjunction",
    "助": "particle",
    "量": "classifier",
    "数": "number",
}

OXFORD_POS = {
    "n.": "noun",
    "v.": "verb",
    "adj.": "adjective",
    "adv.": "adverb",
    "pron.": "pronoun",
    "prep.": "preposition",
    "conj.": "conjunction",
    "det.": "determiner",
    "exclam.": "exclamation",
    "modal v.": "modal verb",
    "auxiliary v.": "auxiliary verb",
    "auxiliary": "auxiliary",
    "article": "article",
    "number": "number",
    "infinitive": "infinitive",
}


def log(message: str) -> None:
    print(message, flush=True)


def clean(text: str | None) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("\u00ad", "")
    text = text.replace("\ufeff", "")
    text = text.replace("\ufffd", "")
    return re.sub(r"\s+", " ", text).strip()


def normalize_unicode(text: str | None) -> str:
    return unicodedata.normalize("NFKC", text or "")


def has_hanzi(text: str) -> bool:
    return bool(HANZI_RE.search(normalize_unicode(text)))


def suppress_pdf_font_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"Could not get FontBBox from font descriptor.*",
    )
    logging.getLogger("pdfminer").setLevel(logging.ERROR)


def open_pdf(path: Path):
    suppress_pdf_font_warnings()
    return pdfplumber.open(path)


def normalize_oxford_pos(text: str) -> str:
    text = clean(text).replace(";", ",")
    text = re.sub(r"[/|]+", ",", text)
    found = []
    for key in sorted(OXFORD_POS, key=len, reverse=True):
        if re.search(rf"(?<![A-Za-z]){re.escape(key)}(?![A-Za-z])", text, re.I):
            value = OXFORD_POS[key]
            if value not in found:
                found.append(value)
    return ", ".join(found)


def looks_like_oxford_word(text: str) -> bool:
    text = clean(text)
    return bool(
        text
        and re.fullmatch(
            r"[A-Za-z][A-Za-z0-9'’\-]*"
            r"(?:\s+[A-Za-z0-9][A-Za-z0-9'’\-]*)*",
            text,
        )
    )


def parse_oxford_cell(cell_text: str, level: str) -> dict[str, Any] | None:
    text = re.sub(r"^[•·]+", "", clean(cell_text))
    if not text:
        return None

    pos_pattern = re.compile(
        r"(?:modal\s+v\.|auxiliary\s+v\.|infinitive|article|number|"
        r"n\.|v\.|adj\.|adv\.|pron\.|prep\.|conj\.|det\.|exclam\.)",
        re.I,
    )
    match = pos_pattern.search(text)

    if match:
        word = clean(text[: match.start()])
        if not looks_like_oxford_word(word):
            return None
        return {
            "id": "",
            "lang": "en",
            "word": word.lower(),
            "pinyin": "",
            "pos": normalize_oxford_pos(text[match.start() :]),
            "definition": "",
            "level": level,
            "source": "",
        }

    if text.lower() in {
        "level",
        "by cefr",
        "words to learn in english",
        "from a1 to b2 level.",
    }:
        return None

    definition = ""
    parenthetical = re.match(r"^(?P<word>.+?)\s+\((?P<definition>.+)\)$", text)
    word = text
    if parenthetical:
        word = clean(parenthetical.group("word"))
        definition = clean(parenthetical.group("definition"))

    if not looks_like_oxford_word(word):
        return None

    return {
        "id": "",
        "lang": "en",
        "word": word.lower(),
        "pinyin": "",
        "pos": "",
        "definition": definition,
        "level": level,
        "source": "",
    }


def get_page_columns(page) -> list[list[dict[str, Any]]]:
    words = page.extract_words(
        x_tolerance=1.5,
        y_tolerance=2,
        keep_blank_chars=False,
    )
    if not words:
        return []

    midpoint = float(page.width) / 2
    left = []
    right = []
    for word in words:
        centre = (float(word["x0"]) + float(word["x1"])) / 2
        (left if centre < midpoint else right).append(word)

    def split_half(items: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        if len(items) < 2:
            return [items] if items else []
        ordered = sorted(items, key=lambda item: float(item["x0"]))
        gaps = []
        for index in range(1, len(ordered)):
            gap = float(ordered[index]["x0"]) - float(ordered[index - 1]["x1"])
            gaps.append((gap, index))
        largest_gap, split_index = max(gaps, key=lambda pair: pair[0])
        if largest_gap <= float(page.width) * 0.02:
            return [ordered]
        return [ordered[:split_index], ordered[split_index:]]

    columns = split_half(left) + split_half(right)
    return [column for column in columns if len(column) >= 2]


def column_lines(words: list[dict[str, Any]]) -> list[str]:
    if not words:
        return []

    ordered = sorted(
        words,
        key=lambda word: (round(float(word["top"]), 1), float(word["x0"])),
    )
    lines: list[list[dict[str, Any]]] = []

    for word in ordered:
        top = float(word["top"])
        target = None
        for line in reversed(lines[-3:]):
            average_top = sum(item["_top"] for item in line) / len(line)
            if abs(top - average_top) <= 3.0:
                target = line
                break
        item = {**word, "_top": top}
        if target is None:
            lines.append([item])
        else:
            target.append(item)

    result = []
    for line in lines:
        line.sort(key=lambda word: float(word["x0"]))
        text = clean(" ".join(word["text"] for word in line))
        if text:
            result.append(text)
    return result


def parse_english(path: Path) -> list[dict[str, Any]]:
    results = []
    current_level = ""

    with open_pdf(path) as pdf:
        for page_number, page in enumerate(pdf.pages, 1):
            page_text = clean(page.extract_text() or "")
            headings = re.findall(r"\b(?:A1|A2|B1|B2|C1)\b", page_text, re.I)
            if headings:
                current_level = headings[-1].upper()

            for column in get_page_columns(page):
                for line in column_lines(column):
                    level_match = OXFORD_LEVEL_RE.fullmatch(line)
                    if level_match:
                        current_level = level_match.group(1).upper()
                        continue
                    if not current_level:
                        continue
                    entry = parse_oxford_cell(line, current_level)
                    if entry:
                        entry["_page"] = page_number
                        results.append(entry)

    unique = {}
    for row in results:
        unique.setdefault(row["word"], row)

    results = list(unique.values())
    for index, row in enumerate(results, 1):
        row["id"] = f"en_{index:05}"
        row["source"] = path.name
        row.pop("_page", None)
    return results


def normalize_zh_pos(text: str) -> str:
    text = clean(text)
    found = []
    for key in sorted(ZH_POS, key=len, reverse=True):
        if key in text and ZH_POS[key] not in found:
            found.append(ZH_POS[key])
    return ", ".join(found)


def parse_hsk_entry_line(line: str, level: str) -> dict[str, Any] | None:
    match = HSK_ENTRY_RE.match(clean(line))
    if not match:
        return None

    word = normalize_unicode(clean(match.group("word")))
    if not has_hanzi(word):
        return None

    rest = clean(match.group("rest"))
    pinyin = clean(match.group("pinyin"))
    pos = ""
    definition = rest

    pos_pattern = re.compile(
        r"^(?P<pos>"
        r"(?:能愿|名|动|形|副|代|介|连|助|量|数)"
        r"(?:\s*[、,]\s*(?:能愿|名|动|形|副|代|介|连|助|量|数))*"
        r"|(?:noun|verb|adjective|adverb|pronoun|preposition|"
        r"conjunction|particle|classifier|number)"
        r"(?:\s*[,、]\s*(?:noun|verb|adjective|adverb|pronoun|"
        r"preposition|conjunction|particle|classifier|number))*"
        r")\s*",
        re.I,
    )
    pos_match = pos_pattern.match(rest)
    if pos_match:
        pos = normalize_zh_pos(pos_match.group("pos")) or clean(pos_match.group("pos"))
        definition = clean(rest[pos_match.end() :])

    return {
        "id": "",
        "lang": "zh",
        "word": word,
        "pinyin": pinyin,
        "pos": pos,
        "definition": definition,
        "level": level,
        "source": "",
        "_number": int(match.group("number")),
    }


def detect_hsk_level(text: str) -> str | None:
    match = HSK_LEVEL_RE.search(normalize_unicode(text))
    if not match:
        return None
    raw = re.sub(r"\s+", "", match.group(1))
    return f"HSK {raw}"


def parse_hsk(path: Path, diagnostic: bool = False) -> list[dict[str, Any]]:
    if diagnostic:
        with open_pdf(path) as pdf:
            log("\n" + "=" * 70)
            log("HSK PDF DIAGNOSTIC — FIRST 5 PAGES")
            for page_number, page in enumerate(pdf.pages[:5], 1):
                log(f"\n--- PAGE {page_number} ---")
                log((page.extract_text() or "")[:5000])
            log("=" * 70)

    results = []
    current_level = None
    pending = None

    with open_pdf(path) as pdf:
        for page_number, page in enumerate(pdf.pages, 1):
            for raw_line in (page.extract_text() or "").splitlines():
                line = clean(raw_line)
                if not line:
                    continue

                level = detect_hsk_level(line)
                if level:
                    if pending:
                        results.append(pending)
                        pending = None
                    current_level = level
                    continue

                if current_level is None:
                    continue

                upper = line.upper()
                if (
                    "NO. WORD PINYIN" in upper
                    or upper == "ENTRIES"
                    or "MANDARINBEAN.COM PAGE" in upper
                    or line.startswith(("⇨", ">>>"))
                ):
                    continue

                parsed = parse_hsk_entry_line(line, current_level)
                if parsed:
                    if pending:
                        results.append(pending)
                    pending = parsed
                    continue

                if pending and not re.fullmatch(r"\d+", line):
                    pending["definition"] = clean(
                        f'{pending["definition"]} {normalize_unicode(line)}'
                    )

    if pending:
        results.append(pending)

    unique = {}
    for row in results:
        row.pop("_number", None)
        existing = unique.get(row["word"])
        if existing is None or len(row["definition"]) > len(existing["definition"]):
            unique[row["word"]] = row

    results = list(unique.values())
    for index, row in enumerate(results, 1):
        row["id"] = f"zh_{index:05}"
        row["source"] = path.name
    return results


def validate_english(rows: list[dict[str, Any]]) -> None:
    counts = defaultdict(int)
    for row in rows:
        counts[row["level"]] += 1

    log("\nOxford CEFR extraction check:")
    for level in ("A1", "A2", "B1", "B2", "C1"):
        log(f" {level}: {counts[level]:,}")
    log(f" TOTAL: {len(rows):,}")

    if len(rows) < 4000:
        raise RuntimeError(
            f"Oxford extraction is too low: {len(rows):,}. "
            "The program will not continue."
        )
    if len(rows) > 5500:
        log("Warning: Oxford extraction is above 5,500; check duplicates.")


def validate_hsk(rows: list[dict[str, Any]], strict: bool = False) -> None:
    counts = defaultdict(int)
    for row in rows:
        counts[row["level"]] += 1

    log("\nHSK extraction check:")
    for level, expected in HSK_EXPECTED.items():
        log(f" {level:<8} {counts[level]:>6,} / expected {expected:,}")
    log(f" TOTAL: {len(rows):,} / expected 11,000")

    if len(rows) < 10000:
        raise RuntimeError(
            f"HSK extraction is too low: {len(rows):,}/11,000. "
            "The program will not continue."
        )
    if strict and len(rows) != 11000:
        raise RuntimeError(
            f"Strict HSK validation failed: {len(rows):,}/11,000 entries."
        )
    if len(rows) != 11000:
        log("Warning: HSK extraction is incomplete; continuing in non-strict mode.")


@lru_cache(maxsize=None)
def wordnet_definition(word: str) -> str:
    try:
        from nltk.corpus import wordnet as wn
        synsets = wn.synsets(word)
        return synsets[0].definition() if synsets else ""
    except Exception:
        return ""


def add_wordnet_definitions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    try:
        import nltk
        from nltk.corpus import wordnet as wn
        try:
            wn.synsets("test")
        except LookupError:
            log("Downloading NLTK WordNet data...")
            nltk.download("wordnet", quiet=False)
            nltk.download("omw-1.4", quiet=False)
            wn.synsets("test")
    except Exception as exc:
        log(f"Warning: WordNet unavailable; definitions skipped: {exc}")
        return rows

    english_rows = [row for row in rows if row["lang"] == "en" and not row["definition"]]
    log(f"Looking up WordNet definitions for {len(english_rows):,} entries...")
    for index, row in enumerate(english_rows, 1):
        definition = wordnet_definition(row["word"])
        if definition:
            row["definition"] = definition
        if index % 500 == 0:
            log(f" WordNet progress: {index:,}/{len(english_rows):,}")
    return rows


_RADICAL_FINDER = None


def get_radical_finder():
    global _RADICAL_FINDER
    if _RADICAL_FINDER is None:
        from cjkradlib import RadicalFinder
        _RADICAL_FINDER = RadicalFinder(lang="zh")
    return _RADICAL_FINDER


@lru_cache(maxsize=None)
def radicals(word: str) -> tuple[str, ...]:
    word = normalize_unicode(word)
    try:
        finder = get_radical_finder()
        found = []
        for char in word:
            if not has_hanzi(char):
                continue
            try:
                result = finder.search(char)
                values = list(getattr(result, "compositions", []) or [])
                found.extend(values if values else [char])
            except Exception:
                found.append(char)
        return tuple(sorted(set(found)))
    except Exception:
        return tuple(sorted(set(char for char in word if has_hanzi(char))))


def prepare_embedding_rows(rows: list[dict[str, Any]]) -> list[str]:
    texts = []
    total = len(rows)
    log(f"Preparing {total:,} embedding texts and radicals...")

    for index, row in enumerate(rows, 1):
        row["radicals"] = list(radicals(row["word"])) if row["lang"] == "zh" else []
        pieces = [row["word"], row["definition"], row["pos"]]
        if row["lang"] == "zh":
            pieces.append(row["pinyin"])
            pieces.append(" ".join(row["radicals"]))
        text = " | ".join(clean(piece) for piece in pieces if clean(piece))
        row["embedding_text"] = text
        texts.append(text)

        if index % 500 == 0 or index == total:
            log(f" Prepared {index:,}/{total:,} entries")
    return texts


def make_graph(rows: list[dict[str, Any]], threshold: float = 0.78, batch_size: int = 64):
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.neighbors import NearestNeighbors
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependencies. Install sentence-transformers, scikit-learn, "
            "pandas, pdfplumber, nltk, and cjkradlib."
        ) from exc

    if not rows:
        return {}, {}, []

    texts = prepare_embedding_rows(rows)

    log("Loading multilingual embedding model...")
    model_start = time.perf_counter()
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    log(f"Model loaded in {time.perf_counter() - model_start:.1f}s")

    log(f"Encoding {len(texts):,} vocabulary entries...")
    encoding_start = time.perf_counter()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=batch_size,
    )
    log(f"Encoding completed in {time.perf_counter() - encoding_start:.1f}s")

    if len(rows) == 1:
        member = dict(rows[0])
        member.pop("embedding_text", None)
        return {
            "concept_00001": {
                "id": "concept_00001",
                "definition": rows[0]["definition"],
                "members": [member],
            }
        }, {}, []

    k = min(30, len(rows))
    log(f"Building nearest-neighbour index with k={k}...")
    neighbour_start = time.perf_counter()
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", n_jobs=-1).fit(vectors)
    distances, indexes = nn.kneighbors(vectors)
    log(f"Nearest-neighbour search completed in {time.perf_counter() - neighbour_start:.1f}s")

    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    review = []
    total_links = len(rows)
    log(f"Processing up to {total_links * (k - 1):,} graph links...")

    for index in range(len(rows)):
        for distance, neighbour in zip(distances[index][1:], indexes[index][1:]):
            score = 1.0 - float(distance)
            neighbour = int(neighbour)
            if score >= threshold:
                union(index, neighbour)
            elif score >= threshold - 0.06 and rows[index]["lang"] != rows[neighbour]["lang"]:
                review.append({
                    "entry_a": rows[index]["id"],
                    "word_a": rows[index]["word"],
                    "entry_b": rows[neighbour]["id"],
                    "word_b": rows[neighbour]["word"],
                    "similarity": round(score, 4),
                })

        if (index + 1) % 1000 == 0 or index + 1 == len(rows):
            log(f" Graph progress: {index + 1:,}/{len(rows):,}")

    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[find(index)].append(row)

    concepts = {}
    radical_index = defaultdict(set)
    log(f"Creating {len(groups):,} concept groups...")

    for number, members in enumerate(groups.values(), 1):
        concept_id = f"concept_{number:05}"
        label = max(members, key=lambda item: len(item["definition"]))
        clean_members = []

        for member in members:
            clean_member = {
                key: member.get(key, "")
                for key in [
                    "id", "lang", "word", "pinyin", "pos", "definition",
                    "level", "radicals",
                ]
            }
            clean_members.append(clean_member)
            for radical in member.get("radicals", []):
                radical_index[radical].add(concept_id)

        concepts[concept_id] = {
            "id": concept_id,
            "definition": label["definition"],
            "members": clean_members,
        }

    return concepts, {key: sorted(value) for key, value in radical_index.items()}, review


def resolve_pdf(requested: str, label: str) -> Path:
    path = Path(requested).expanduser()
    candidates = [path] if path.is_absolute() else [
        Path.cwd() / path,
        Path(__file__).resolve().parent / path,
        Path.home() / "Downloads" / path.name,
        Path.home() / "Desktop" / path.name,
        Path.home() / "Documents" / path.name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    checked = "\n".join(f" - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        f"{label} PDF not found.\nRequested: {requested}\nChecked:\n{checked}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a bilingual English-Chinese vocabulary semantic concept graph."
    )
    parser.add_argument("--english", default="ENGLISH.pdf")
    parser.add_argument("--hsk", default="HSK.pdf")
    parser.add_argument("--out", default="output")
    parser.add_argument("--threshold", type=float, default=0.78)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--strict-hsk", action="store_true")
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args()

    log("=" * 70)
    log("VOCABULARY NODE BUILDER — PERFORMANCE-IMPROVED")
    log("=" * 70)

    english_path = resolve_pdf(args.english, "English")
    hsk_path = resolve_pdf(args.hsk, "HSK")
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)

    log(f"English PDF: {english_path}")
    log(f"HSK PDF: {hsk_path}")

    log("\n[1/5] Parsing Oxford 5000...")
    english = parse_english(english_path)
    log(f"Extracted: {len(english):,}")
    validate_english(english)

    log("\n[2/5] Parsing HSK 1–9...")
    chinese = parse_hsk(hsk_path, diagnostic=args.diagnostic)
    log(f"Extracted: {len(chinese):,}")
    validate_hsk(chinese, strict=args.strict_hsk)

    log("\n[3/5] Adding English WordNet definitions...")
    english = add_wordnet_definitions(english)
    rows = english + chinese
    log(f"TOTAL VOCABULARY: {len(rows):,}")

    log("\n[4/5] Building semantic graph...")
    concepts, radical_index, review = make_graph(
        rows,
        threshold=args.threshold,
        batch_size=args.batch_size,
    )

    log("\n[5/5] Writing output...")
    pd.DataFrame(rows).drop(columns=["embedding_text"], errors="ignore").to_csv(
        output / "entries.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (output / "concepts.json").write_text(
        json.dumps(concepts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "radicals.json").write_text(
        json.dumps(radical_index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(review).drop_duplicates().to_csv(
        output / "review.csv",
        index=False,
        encoding="utf-8-sig",
    )

    log("\n" + "=" * 70)
    log("SUCCESS")
    log("=" * 70)
    log(f"Oxford 5000: {len(english):,}")
    log(f"HSK 1–9: {len(chinese):,}")
    log(f"TOTAL: {len(rows):,}")
    log(f"CONCEPTS: {len(concepts):,}")
    log(f"OUTPUT: {output}")
    log("=" * 70)


if __name__ == "__main__":
    main()
