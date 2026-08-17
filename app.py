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
    r"(?:NEW\s+HSK\s+VOCABULARY\s*)?LEVEL\s*([1-9](?:\s*-\s*[1-9])?)",
    re.I,
)
HSK_ENTRY_RE = re.compile(
    r"^(?P<number>\d+)\s+(?P<word>\S+)\s+(?P<pinyin>\S+)(?:\s+(?P<rest>.*))?$"
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
    "能愿": "modal", "名": "noun", "动": "verb", "形": "adjective",
    "副": "adverb", "代": "pronoun", "介": "preposition",
    "连": "conjunction", "助": "particle", "量": "classifier", "数": "number",
}

OXFORD_POS = {
    "n.": "noun", "v.": "verb", "adj.": "adjective", "adv.": "adverb",
    "pron.": "pronoun", "prep.": "preposition", "conj.": "conjunction",
    "det.": "determiner", "exclam.": "exclamation", "modal v.": "modal verb",
    "auxiliary v.": "auxiliary verb", "auxiliary": "auxiliary", "article": "article",
    "number": "number", "infinitive": "infinitive",
}


def log(message: str) -> None:
    print(message, flush=True)


def clean(text: str | None) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("\u00ad", "").replace("\ufeff", "").replace("\ufffd", "")
    return re.sub(r"\s+", " ", text).strip()


def normalize_unicode(text: str | None) -> str:
    return unicodedata.normalize("NFKC", text or "")


def has_hanzi(text: str) -> bool:
    return bool(HANZI_RE.search(normalize_unicode(text)))


def suppress_pdf_font_warnings() -> None:
    warnings.filterwarnings("ignore", message=r"Could not get FontBBox from font descriptor.*")
    logging.getLogger("pdfminer").setLevel(logging.ERROR)


def open_pdf(path: Path):
    suppress_pdf_font_warnings()
    return pdfplumber.open(path)


def normalize_oxford_pos(text: str) -> str:
    text = re.sub(r"[/|;]+", ",", clean(text))
    found = []
    for key in sorted(OXFORD_POS, key=len, reverse=True):
        if re.search(rf"(?<![A-Za-z]){re.escape(key)}(?![A-Za-z])", text, re.I):
            if OXFORD_POS[key] not in found:
                found.append(OXFORD_POS[key])
    return ", ".join(found)


def looks_like_oxford_word(text: str) -> bool:
    return bool(re.fullmatch(
        r"[A-Za-z][A-Za-z0-9'’\-]*(?:\s+[A-Za-z0-9][A-Za-z0-9'’\-]*)*", clean(text)
    ))


def parse_oxford_cell(cell_text: str, level: str) -> dict[str, Any] | None:
    text = re.sub(r"^[•·]+", "", clean(cell_text))
    if not text:
        return None
    pos_pattern = re.compile(
        r"(?:modal\s+v\.|auxiliary\s+v\.|infinitive|article|number|n\.|v\.|adj\.|adv\.|pron\.|prep\.|conj\.|det\.|exclam\.)",
        re.I,
    )
    match = pos_pattern.search(text)
    if match:
        word = clean(text[:match.start()])
        if not looks_like_oxford_word(word):
            return None
        return {"id": "", "lang": "en", "word": word.lower(), "pinyin": "",
                "pos": normalize_oxford_pos(text[match.start():]), "definition": "",
                "level": level, "source": ""}
    if text.lower() in {"level", "by cefr", "words to learn in english", "from a1 to b2 level."}:
        return None
    definition = ""
    parenthetical = re.match(r"^(?P<word>.+?)\s+\((?P<definition>.+)\)$", text)
    word = text
    if parenthetical:
        word = clean(parenthetical.group("word"))
        definition = clean(parenthetical.group("definition"))
    if not looks_like_oxford_word(word):
        return None
    return {"id": "", "lang": "en", "word": word.lower(), "pinyin": "", "pos": "",
            "definition": definition, "level": level, "source": ""}


def get_page_columns(page) -> list[list[dict[str, Any]]]:
    words = page.extract_words(x_tolerance=1.5, y_tolerance=2, keep_blank_chars=False)
    if not words:
        return []
    midpoint = float(page.width) / 2
    left = [w for w in words if (float(w["x0"]) + float(w["x1"])) / 2 < midpoint]
    right = [w for w in words if (float(w["x0"]) + float(w["x1"])) / 2 >= midpoint]

    def split_half(items):
        if len(items) < 2:
            return [items] if items else []
        ordered = sorted(items, key=lambda item: float(item["x0"]))
        gaps = [(float(ordered[i]["x0"]) - float(ordered[i - 1]["x1"]), i)
                for i in range(1, len(ordered))]
        largest_gap, split_index = max(gaps, key=lambda item: item[0])
        if largest_gap <= float(page.width) * 0.02:
            return [ordered]
        return [ordered[:split_index], ordered[split_index:]]

    return [column for column in split_half(left) + split_half(right) if len(column) >= 2]


def column_lines(words) -> list[str]:
    if not words:
        return []
    ordered = sorted(words, key=lambda w: (round(float(w["top"]), 1), float(w["x0"])))
    lines = []
    for word in ordered:
        top = float(word["top"])
        target = None
        for line in reversed(lines[-3:]):
            if abs(top - sum(item["_top"] for item in line) / len(line)) <= 3:
                target = line
                break
        item = {**word, "_top": top}
        (target if target is not None else lines.setdefault(len(lines), [])).append(item)
    return [clean(" ".join(w["text"] for w in sorted(line, key=lambda x: float(x["x0"]))))
            for line in lines if line]


def parse_english(path: Path) -> list[dict[str, Any]]:
    results, current_level = [], ""
    with open_pdf(path) as pdf:
        for page_number, page in enumerate(pdf.pages, 1):
            headings = re.findall(r"\b(?:A1|A2|B1|B2|C1)\b", clean(page.extract_text() or ""), re.I)
            if headings:
                current_level = headings[-1].upper()
            for column in get_page_columns(page):
                for line in column_lines(column):
                    match = OXFORD_LEVEL_RE.fullmatch(line)
                    if match:
                        current_level = match.group(1).upper()
                        continue
                    if current_level:
                        entry = parse_oxford_cell(line, current_level)
                        if entry:
                            results.append(entry)
    unique = {}
    for row in results:
        unique.setdefault(row["word"], row)
    results = list(unique.values())
    for i, row in enumerate(results, 1):
        row["id"], row["source"] = f"en_{i:05}", path.name
    return results


def normalize_zh_pos(text: str) -> str:
    found = []
    for key in sorted(ZH_POS, key=len, reverse=True):
        if key in clean(text) and ZH_POS[key] not in found:
            found.append(ZH_POS[key])
    return ", ".join(found)


def parse_hsk_entry_line(line: str, level: str):
    match = HSK_ENTRY_RE.match(clean(line))
    if not match:
        return None
    word = normalize_unicode(match.group("word"))
    if not has_hanzi(word):
        return None
    rest = clean(match.group("rest"))
    pos, definition = "", rest
    pos_pattern = re.compile(
        r"^(?P<pos>(?:能愿|名|动|形|副|代|介|连|助|量|数)(?:\s*[、,]\s*(?:能愿|名|动|形|副|代|介|连|助|量|数))*|(?:noun|verb|adjective|adverb|pronoun|preposition|conjunction|particle|classifier|number)(?:\s*[,、]\s*(?:noun|verb|adjective|adverb|pronoun|preposition|conjunction|particle|classifier|number))*)\s*",
        re.I,
    )
    pos_match = pos_pattern.match(rest)
    if pos_match:
        pos = normalize_zh_pos(pos_match.group("pos")) or clean(pos_match.group("pos"))
        definition = clean(rest[pos_match.end():])
    return {"id": "", "lang": "zh", "word": word, "pinyin": clean(match.group("pinyin")),
            "pos": pos, "definition": definition, "level": level, "source": "",
            "_number": int(match.group("number"))}


def detect_hsk_level(text: str) -> str | None:
    match = HSK_LEVEL_RE.search(normalize_unicode(text))
    return f"HSK {re.sub(r'\s+', '', match.group(1))}" if match else None


def parse_hsk(path: Path, diagnostic: bool = False) -> list[dict[str, Any]]:
    results, current_level, pending = [], None, None
    if diagnostic:
        with open_pdf(path) as pdf:
            for number, page in enumerate(pdf.pages[:5], 1):
                log(f"--- PAGE {number} ---")
                log((page.extract_text() or "")[:5000])
    with open_pdf(path) as pdf:
        for page in pdf.pages:
            for raw_line in (page.extract_text() or "").splitlines():
                line = clean(raw_line)
                if not line:
                    continue
                level = detect_hsk_level(line)
                if level:
                    if pending:
                        results.append(pending)
                    pending, current_level = None, level
                    continue
                if current_level is None:
                    continue
                upper = line.upper()
                if ("NO. WORD PINYIN" in upper or upper == "ENTRIES" or
                        "MANDARINBEAN.COM PAGE" in upper or line.startswith(("⇨", ">>>"))):
                    continue
                parsed = parse_hsk_entry_line(line, current_level)
                if parsed:
                    if pending:
                        results.append(pending)
                    pending = parsed
                elif pending and not re.fullmatch(r"\d+", line):
                    pending["definition"] = clean(f'{pending["definition"]} {line}')
    if pending:
        results.append(pending)
    unique = {}
    for row in results:
        row.pop("_number", None)
        if row["word"] not in unique or len(row["definition"]) > len(unique[row["word"]]["definition"]):
            unique[row["word"]] = row
    results = list(unique.values())
    for i, row in enumerate(results, 1):
        row["id"], row["source"] = f"zh_{i:05}", path.name
    return results


def validate_english(rows):
    counts = defaultdict(int)
    for row in rows:
        counts[row["level"]] += 1
    log("\nOxford CEFR extraction check:")
    for level in ("A1", "A2", "B1", "B2", "C1"):
        log(f" {level}: {counts[level]:,}")
    log(f" TOTAL: {len(rows):,}")
    if len(rows) < 4000:
        raise RuntimeError(f"Oxford extraction too low: {len(rows):,}")


def validate_hsk(rows, strict=False):
    counts = defaultdict(int)
    for row in rows:
        counts[row["level"]] += 1
    log("\nHSK extraction check:")
    for level, expected in HSK_EXPECTED.items():
        log(f" {level:<8} {counts[level]:>6,} / expected {expected:,}")
    log(f" TOTAL: {len(rows):,} / expected 11,000")
    if len(rows) < 10000:
        raise RuntimeError(f"HSK extraction too low: {len(rows):,}/11,000")
    if strict and len(rows) != 11000:
        raise RuntimeError(f"Strict HSK validation failed: {len(rows):,}/11,000")
    if len(rows) != 11000:
        log("Warning: continuing with incomplete HSK extraction.")


@lru_cache(maxsize=None)
def wordnet_definition(word: str) -> str:
    try:
        from nltk.corpus import wordnet as wn
        synsets = wn.synsets(word)
        return synsets[0].definition() if synsets else ""
    except Exception:
        return ""


def add_wordnet_definitions(rows):
    try:
        import nltk
        from nltk.corpus import wordnet as wn
        try:
            wn.synsets("test")
        except LookupError:
            nltk.download("wordnet", quiet=False)
            nltk.download("omw-1.4", quiet=False)
            wn.synsets("test")
    except Exception as exc:
        log(f"Warning: WordNet unavailable: {exc}")
        return rows
    candidates = [r for r in rows if r["lang"] == "en" and not r["definition"]]
    for i, row in enumerate(candidates, 1):
        row["definition"] = wordnet_definition(row["word"])
        if i % 500 == 0:
            log(f" WordNet progress: {i:,}/{len(candidates):,}")
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
    try:
        finder = get_radical_finder()
        found = []
        for char in normalize_unicode(word):
            if has_hanzi(char):
                try:
                    result = finder.search(char)
                    values = list(getattr(result, "compositions", []) or [])
                    found.extend(values if values else [char])
                except Exception:
                    found.append(char)
        return tuple(sorted(set(found)))
    except Exception:
        return tuple(sorted(set(char for char in word if has_hanzi(char))))


def prepare_embedding_rows(rows):
    texts = []
    for i, row in enumerate(rows, 1):
        row["radicals"] = list(radicals(row["word"])) if row["lang"] == "zh" else []
        pieces = [row["word"], row["definition"], row["pos"]]
        if row["lang"] == "zh":
            pieces.extend([row["pinyin"], " ".join(row["radicals"])])
        row["embedding_text"] = " | ".join(clean(p) for p in pieces if clean(p))
        texts.append(row["embedding_text"])
        if i % 500 == 0 or i == len(rows):
            log(f" Prepared {i:,}/{len(rows):,} entries")
    return texts


def make_graph(rows, threshold=0.78, batch_size=32, mutual_only=True, min_group_similarity=0.80):
    from sentence_transformers import SentenceTransformer
    from sklearn.neighbors import NearestNeighbors

    if not rows:
        return {}, {}, []

    texts = prepare_embedding_rows(rows)
    log("Loading multilingual embedding model...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    log(f"Encoding {len(texts):,} vocabulary entries...")
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True, batch_size=batch_size)

    if len(rows) == 1:
        return {"concept_00001": {"id": "concept_00001", "definition": rows[0]["definition"], "members": [rows[0]]}}, {}, []

    k = min(30, len(rows))
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", n_jobs=-1).fit(vectors)
    distances, indexes = nn.kneighbors(vectors)
    neighbour_sets = {i: set(map(int, indexes[i][1:])) for i in range(len(rows))}

    parent = list(range(len(rows)))
    rank = [0] * len(rows)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a == b:
            return
        if rank[a] < rank[b]:
            a, b = b, a
        parent[b] = a
        if rank[a] == rank[b]:
            rank[a] += 1

    review = []
    log("Building constrained concept groups...")
    for i in range(len(rows)):
        for distance, j_value in zip(distances[i][1:], indexes[i][1:]):
            j = int(j_value)
            score = 1.0 - float(distance)
            mutual = i in neighbour_sets[j]
            cross_language = rows[i]["lang"] != rows[j]["lang"]
            compatible_pos = not (rows[i].get("pos") and rows[j].get("pos") and rows[i]["pos"] != rows[j]["pos"])

            if score >= threshold and (not mutual_only or mutual) and compatible_pos:
                union(i, j)
            elif cross_language and score >= threshold - 0.06:
                review.append({
                    "entry_a": rows[i]["id"], "word_a": rows[i]["word"],
                    "entry_b": rows[j]["id"], "word_b": rows[j]["word"],
                    "similarity": round(score, 4), "mutual": mutual,
                    "compatible_pos": compatible_pos,
                })
        if (i + 1) % 1000 == 0 or i + 1 == len(rows):
            log(f" Graph progress: {i + 1:,}/{len(rows):,}")

    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)

    concepts = {}
    radical_index = defaultdict(set)
    for number, members in enumerate(groups.values(), 1):
        concept_id = f"concept_{number:05}"
        label = max(members, key=lambda item: len(item.get("definition", "")))
        clean_members = []
        for member in members:
            clean_member = {key: member.get(key, "") for key in [
                "id", "lang", "word", "pinyin", "pos", "definition", "level", "radicals"
            ]}
            clean_members.append(clean_member)
            for radical in member.get("radicals", []):
                radical_index[radical].add(concept_id)
        concepts[concept_id] = {"id": concept_id, "definition": label["definition"], "members": clean_members}

    return concepts, {key: sorted(value) for key, value in radical_index.items()}, review


def resolve_pdf(requested: str, label: str) -> Path:
    path = Path(requested).expanduser()
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, Path(__file__).resolve().parent / path,
        Path.home() / "Downloads" / path.name, Path.home() / "Desktop" / path.name, Path.home() / "Documents" / path.name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"{label} PDF not found: {requested}\nChecked:\n" + "\n".join(map(str, candidates)))


def main():
    parser = argparse.ArgumentParser(description="Build a bilingual vocabulary semantic graph.")
    parser.add_argument("--english", default="ENGLISH.pdf")
    parser.add_argument("--hsk", default="HSK.pdf")
    parser.add_argument("--out", default="output")
    parser.add_argument("--threshold", type=float, default=0.78)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--strict-hsk", action="store_true")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--allow-chaining", action="store_true")
    args = parser.parse_args()

    log("=" * 70)
    log("VOCABULARY NODE BUILDER — CONSTRAINED GRAPH")
    log("=" * 70)
    english_path = resolve_pdf(args.english, "English")
    hsk_path = resolve_pdf(args.hsk, "HSK")
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)

    log("\n[1/5] Parsing Oxford 5000...")
    english = parse_english(english_path)
    validate_english(english)
    log(f"Extracted: {len(english):,}")

    log("\n[2/5] Parsing HSK 1–9...")
    chinese = parse_hsk(hsk_path, diagnostic=args.diagnostic)
    validate_hsk(chinese, strict=args.strict_hsk)
    log(f"Extracted: {len(chinese):,}")

    log("\n[3/5] Adding English WordNet definitions...")
    rows = add_wordnet_definitions(english + chinese)
    log(f"TOTAL VOCABULARY: {len(rows):,}")

    log("\n[4/5] Building semantic graph...")
    concepts, radical_index, review = make_graph(
        rows,
        threshold=args.threshold,
        batch_size=args.batch_size,
        mutual_only=not args.allow_chaining,
    )

    log("\n[5/5] Writing output...")
    pd.DataFrame(rows).drop(columns=["embedding_text"], errors="ignore").to_csv(output / "entries.csv", index=False, encoding="utf-8-sig")
    (output / "concepts.json").write_text(json.dumps(concepts, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "radicals.json").write_text(json.dumps(radical_index, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(review).drop_duplicates().to_csv(output / "review.csv", index=False, encoding="utf-8-sig")
    log(f"SUCCESS | concepts={len(concepts):,} | output={output}")


if __name__ == "__main__":
    main()
