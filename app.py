from __future__ import annotations

import argparse
import json
import logging
import re
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
    r"^(?P<number>\d+)\s+(?P<word>\S+)\s+(?P<pinyin>\S+)"
    r"(?:\s+(?P<rest>.*))?$"
)

HSK_EXPECTED = {
    "HSK 1": 300, "HSK 2": 200, "HSK 3": 500,
    "HSK 4": 1000, "HSK 5": 1600, "HSK 6": 1800,
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

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

NUMBER_WORDS = {
    "zero": {"零", "〇"}, "one": {"一"}, "two": {"二", "两"},
    "three": {"三"}, "four": {"四"}, "five": {"五"}, "six": {"六"},
    "seven": {"七"}, "eight": {"八"}, "nine": {"九"}, "ten": {"十"},
    "hundred": {"百"}, "thousand": {"千"},
}

FUNCTION_WORDS = {
    "a", "an", "and", "as", "because", "but", "if", "nor", "or",
    "than", "that", "the", "though", "unless", "until", "when",
    "where", "whether", "while", "how", "why", "also", "again",
}


def log(message: str) -> None:
    print(message, flush=True)


def clean(text: str | None) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("\u00ad", "").replace("\ufeff", "").replace("\ufffd", "")
    return re.sub(r"\s+", " ", text).strip()


def clean_definition(text: str | None) -> str:
    text = clean(text)
    text = re.sub(r"^[、，,;:|]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_unicode(text: str | None) -> str:
    return unicodedata.normalize("NFKC", text or "")


def has_hanzi(text: str) -> bool:
    return bool(HANZI_RE.search(normalize_unicode(text)))


def remove_tones(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text or "")
    return "".join(
        char for char in decomposed
        if unicodedata.category(char) != "Mn"
    ).lower()


def open_pdf(path: Path):
    warnings.filterwarnings(
        "ignore",
        message=r"Could not get FontBBox from font descriptor.*",
    )
    logging.getLogger("pdfminer").setLevel(logging.ERROR)
    return pdfplumber.open(path)


def normalize_oxford_pos(text: str) -> str:
    text = re.sub(r"[/|;]+", ",", clean(text))
    found = []
    for key in sorted(OXFORD_POS, key=len, reverse=True):
        if re.search(rf"(?<![A-Za-z]){re.escape(key)}(?![A-Za-z])", text, re.I):
            value = OXFORD_POS[key]
            if value not in found:
                found.append(value)
    return ", ".join(found)


def looks_like_oxford_word(text: str) -> bool:
    return bool(re.fullmatch(
        r"[A-Za-z][A-Za-z0-9'’\-]*(?:\s+[A-Za-z0-9][A-Za-z0-9'’\-]*)*",
        clean(text),
    ))


def parse_oxford_cell(text: str, level: str):
    text = re.sub(r"^[•·]+", "", clean(text))
    if not text:
        return None

    pattern = re.compile(
        r"(?:modal\s+v\.|auxiliary\s+v\.|infinitive|article|number|"
        r"n\.|v\.|adj\.|adv\.|pron\.|prep\.|conj\.|det\.|exclam\.)",
        re.I,
    )
    match = pattern.search(text)
    if match:
        word = clean(text[:match.start()])
        if not looks_like_oxford_word(word):
            return None
        return {
            "id": "", "lang": "en", "word": word.lower(), "pinyin": "",
            "pos": normalize_oxford_pos(text[match.start():]),
            "definition": "", "level": level, "source": "",
        }

    if text.lower() in {
        "level", "by cefr", "words to learn in english",
        "from a1 to b2 level.",
    }:
        return None

    definition = ""
    parenthetical = re.match(
        r"^(?P<word>.+?)\s+\((?P<definition>.+)\)$",
        text,
    )
    word = text
    if parenthetical:
        word = clean(parenthetical.group("word"))
        definition = clean_definition(parenthetical.group("definition"))

    if not looks_like_oxford_word(word):
        return None

    return {
        "id": "", "lang": "en", "word": word.lower(), "pinyin": "",
        "pos": "", "definition": definition, "level": level, "source": "",
    }


def get_page_columns(page):
    words = page.extract_words(
        x_tolerance=1.5,
        y_tolerance=2,
        keep_blank_chars=False,
    )
    if not words:
        return []

    midpoint = float(page.width) / 2
    left = [
        word for word in words
        if (float(word["x0"]) + float(word["x1"])) / 2 < midpoint
    ]
    right = [
        word for word in words
        if (float(word["x0"]) + float(word["x1"])) / 2 >= midpoint
    ]

    def split_half(items):
        if len(items) < 2:
            return [items] if items else []
        ordered = sorted(items, key=lambda item: float(item["x0"]))
        gaps = [
            (float(ordered[i]["x0"]) - float(ordered[i - 1]["x1"]), i)
            for i in range(1, len(ordered))
        ]
        largest_gap, split_index = max(gaps, key=lambda item: item[0])
        if largest_gap <= float(page.width) * 0.02:
            return [ordered]
        return [ordered[:split_index], ordered[split_index:]]

    return [
        column
        for column in split_half(left) + split_half(right)
        if len(column) >= 2
    ]


def column_lines(words):
    if not words:
        return []

    ordered = sorted(
        words,
        key=lambda word: (
            round(float(word["top"]), 1),
            float(word["x0"]),
        ),
    )
    lines = []

    for word in ordered:
        top = float(word["top"])
        target = None
        for line in reversed(lines[-3:]):
            average_top = sum(item["_top"] for item in line) / len(line)
            if abs(top - average_top) <= 3:
                target = line
                break
        item = {**word, "_top": top}
        if target is None:
            lines.append([item])
        else:
            target.append(item)

    return [
        clean(" ".join(
            word["text"]
            for word in sorted(line, key=lambda item: float(item["x0"]))
        ))
        for line in lines
        if line
    ]


def parse_english(path: Path):
    results = []
    current_level = ""

    with open_pdf(path) as pdf:
        for page in pdf.pages:
            headings = re.findall(
                r"\b(?:A1|A2|B1|B2|C1)\b",
                clean(page.extract_text() or ""),
                re.I,
            )
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
    for index, row in enumerate(results, 1):
        row["id"] = f"en_{index:05}"
        row["source"] = path.name
        row["definition"] = clean_definition(row.get("definition", ""))
    return results


def normalize_zh_pos(text: str) -> str:
    found = []
    for key in sorted(ZH_POS, key=len, reverse=True):
        value = ZH_POS[key]
        if key in clean(text) and value not in found:
            found.append(value)
    return ", ".join(found)


def parse_hsk_entry_line(line: str, level: str):
    match = HSK_ENTRY_RE.match(clean(line))
    if not match:
        return None

    word = normalize_unicode(match.group("word"))
    if not has_hanzi(word):
        return None

    rest = clean(match.group("rest"))
    pos = ""
    definition = rest
    pattern = re.compile(
        r"^(?P<pos>(?:能愿|名|动|形|副|代|介|连|助|量|数)"
        r"(?:\s*[、,]\s*(?:能愿|名|动|形|副|代|介|连|助|量|数))*"
        r"|(?:noun|verb|adjective|adverb|pronoun|preposition|"
        r"conjunction|particle|classifier|number)"
        r"(?:\s*[,、]\s*(?:noun|verb|adjective|adverb|pronoun|"
        r"preposition|conjunction|particle|classifier|number))*)\s*",
        re.I,
    )
    pos_match = pattern.match(rest)
    if pos_match:
        pos = normalize_zh_pos(pos_match.group("pos")) or clean(pos_match.group("pos"))
        definition = clean_definition(rest[pos_match.end():])

    return {
        "id": "", "lang": "zh", "word": word,
        "pinyin": clean(match.group("pinyin")), "pos": pos,
        "definition": definition, "level": level, "source": "",
        "_number": int(match.group("number")),
    }


def detect_hsk_level(text: str):
    match = HSK_LEVEL_RE.search(normalize_unicode(text))
    if not match:
        return None
    level_text = re.sub(r"\s+", "", match.group(1))
    return f"HSK {level_text}"


def parse_hsk(path: Path, diagnostic=False):
    results = []
    current_level = None
    pending = None

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
                elif pending and not re.fullmatch(r"\d+", line):
                    pending["definition"] = clean_definition(
                        f'{pending["definition"]} {line}'
                    )

    if pending:
        results.append(pending)

    unique = {}
    for row in results:
        row.pop("_number", None)
        row["definition"] = clean_definition(row.get("definition", ""))
        if (
            row["word"] not in unique
            or len(row["definition"]) > len(unique[row["word"]]["definition"])
        ):
            unique[row["word"]] = row

    results = list(unique.values())
    for index, row in enumerate(results, 1):
        row["id"] = f"zh_{index:05}"
        row["source"] = path.name
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
def wordnet_definitions(word: str) -> tuple[str, ...]:
    try:
        from nltk.corpus import wordnet as wn
        return tuple(dict.fromkeys(
            clean_definition(synset.definition())
            for synset in wn.synsets(word)
            if clean_definition(synset.definition())
        ))
    except Exception:
        return ()


def add_wordnet_definitions(rows, max_senses=5):
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

    candidates = [
        row for row in rows
        if row["lang"] == "en" and not row["definition"]
    ]
    for index, row in enumerate(candidates, 1):
        senses = wordnet_definitions(row["word"])[:max_senses]
        if senses:
            row["definitions"] = list(senses)
            row["definition"] = " ; ".join(senses)
        else:
            row["definitions"] = []
        if index % 500 == 0:
            log(f" WordNet progress: {index:,}/{len(candidates):,}")
    for row in rows:
        row.setdefault("definitions", [row["definition"]] if row["definition"] else [])
    return rows


_RADICAL_FINDER = None


def get_radical_finder():
    global _RADICAL_FINDER
    if _RADICAL_FINDER is None:
        from cjkradlib import RadicalFinder
        _RADICAL_FINDER = RadicalFinder(lang="zh")
    return _RADICAL_FINDER

@lru_cache(maxsize=None)
def radicals(word: str):
    try:
        finder = get_radical_finder()
        found = []

        for char in normalize_unicode(word):
            if has_hanzi(char):
                try:
                    result = finder.search(char)
                    values = list(
                        getattr(result, "compositions", []) or []
                    )
                    found.extend(
                        values if values else [char]
                    )
                except Exception:
                    found.append(char)

        return tuple(sorted(set(found)))

    except Exception:
        fallback = set()

        for char in word:
            if has_hanzi(char):
                fallback.add(char)

        return tuple(sorted(fallback))
        
def prepare_embeddings(rows):
    texts = []
    for index, row in enumerate(rows, 1):
        row["definition"] = clean_definition(row.get("definition", ""))
        row["radicals"] = list(radicals(row["word"])) if row["lang"] == "zh" else []
        pieces = [row["word"], row["definition"], row["pos"]]
        if row["lang"] == "zh":
            pieces.append(row["pinyin"])
        row["embedding_text"] = " | ".join(clean(p) for p in pieces if clean(p))
        texts.append(row["embedding_text"])
        if index % 500 == 0 or index == len(rows):
            log(f" Prepared {index:,}/{len(rows):,} entries")
    return texts


def pos_set(value: str) -> set[str]:
    return {part.strip() for part in value.split(",") if part.strip()}


def pos_relation(row_a, row_b):
    first = pos_set(row_a.get("pos", ""))
    second = pos_set(row_b.get("pos", ""))
    if not first or not second:
        return "unknown"
    if first & second:
        return "match"
    return "conflict"


def lexical_features(row_a, row_b):
    word_a = row_a["word"].lower()
    word_b = row_b["word"].lower()
    stem_match = False
    if row_a["lang"] == row_b["lang"] == "en":
        stem_a = re.sub(r"(ness|ment|tion|sion|ing|ed|er|ly|s)$", "", word_a)
        stem_b = re.sub(r"(ness|ment|tion|sion|ing|ed|er|ly|s)$", "", word_b)
        stem_match = len(stem_a) >= 4 and stem_a == stem_b
    shared_radicals = sorted(set(row_a.get("radicals", [])) & set(row_b.get("radicals", [])))
    return {"stem_match": stem_match, "shared_radicals": shared_radicals}


def translation_confidence(similarity, english_row, chinese_row, mutual, relation):
    score = 0.80 * similarity
    if mutual:
        score += 0.06
    if relation == "match":
        score += 0.09
    elif relation == "conflict":
        score -= 0.20
    if english_row.get("definition") and chinese_row.get("definition"):
        score += 0.05
    return round(max(0.0, min(score, 1.0)), 4)


def semantic_confidence(similarity, mutual, relation, features):
    score = 0.70 * similarity
    if mutual:
        score += 0.10
    if relation == "match":
        score += 0.10
    elif relation == "conflict":
        score -= 0.15
    if features["stem_match"]:
        score += 0.10
    return round(max(0.0, min(score, 1.0)), 4)


def build_nodes(rows):
    return [{key: row.get(key, "") for key in [
        "id", "lang", "word", "pinyin", "pos", "definition", "definitions",
        "level", "source", "radicals",
    ]} for row in rows]


def iter_top_neighbours(query_vectors, reference_vectors, query_rows, reference_rows, k):
    from sklearn.neighbors import NearestNeighbors
    index = NearestNeighbors(
        n_neighbors=min(k, len(reference_rows)),
        metric="cosine",
        n_jobs=-1,
    ).fit(reference_vectors)
    distances, indexes = index.kneighbors(query_vectors)
    for query_pos, (distance_row, index_row) in enumerate(zip(distances, indexes)):
        for distance, reference_pos in zip(distance_row, index_row):
            yield query_rows[query_pos], reference_rows[int(reference_pos)], float(distance)


def build_same_language_edges(rows, vectors, indices_by_lang, threshold, k):
    semantic_edges = []
    morphology_edges = []
    for language, indices in indices_by_lang.items():
        language_rows = [rows[i] for i in indices]
        language_vectors = vectors[indices]
        neighbour_pairs = list(iter_top_neighbours(
            language_vectors, language_vectors, language_rows, language_rows, k + 1
        ))
        neighbour_map = defaultdict(set)
        for source, target, distance in neighbour_pairs:
            if source["id"] != target["id"]:
                neighbour_map[source["id"]].add(target["id"])
        seen = set()
        for source, target, distance in neighbour_pairs:
            if source["id"] == target["id"]:
                continue
            pair = tuple(sorted((source["id"], target["id"])))
            if pair in seen:
                continue
            seen.add(pair)
            similarity = round(1.0 - distance, 4)
            relation = pos_relation(source, target)
            features = lexical_features(source, target)
            mutual = target["id"] in neighbour_map[source["id"]] and source["id"] in neighbour_map[target["id"]]
            confidence = semantic_confidence(similarity, mutual, relation, features)
            record = {
                "source": source["id"], "target": target["id"],
                "source_word": source["word"], "target_word": target["word"],
                "language": language, "relationship": "semantic_neighbour",
                "similarity": similarity, "confidence": confidence,
                "mutual_neighbour": mutual, "pos_relation": relation,
            }
            if similarity >= threshold and mutual and relation != "conflict":
                semantic_edges.append(record)
            if language == "en" and features["stem_match"]:
                morphology_edges.append({**record, "relationship": "morphological", "confidence": max(confidence, 0.90)})
    return semantic_edges, morphology_edges


def has_definition(row):
    return bool(clean(row.get("definition", "")))


def numeric_mismatch(english_word, chinese_word):
    expected = NUMBER_WORDS.get(english_word.lower())
    return expected is not None and chinese_word not in expected


def translation_quality(confidence, similarity):
    if confidence >= 0.88 and similarity >= 0.80:
        return "strong"
    if confidence >= 0.80 and similarity >= 0.74:
        return "probable"
    if confidence >= 0.70:
        return "review"
    return "reject"


def translation_decision(similarity, confidence, english_row, chinese_row, relation):
    english_word = english_row["word"].lower()
    chinese_word = chinese_row["word"]
    chinese_has_definition = has_definition(chinese_row)
    english_has_definition = has_definition(english_row)

    if numeric_mismatch(english_word, chinese_word):
        return "review"
    if relation == "conflict":
        return "review"
    if english_word in FUNCTION_WORDS and not chinese_has_definition:
        return "review"
    if not chinese_has_definition and similarity < 0.90:
        return "review"
    if confidence >= 0.84 and similarity >= 0.78 and english_has_definition and chinese_has_definition:
        return "strong"
    if confidence >= 0.80 and similarity >= 0.74 and chinese_has_definition:
        return "probable"
    if confidence >= 0.70:
        return "review"
    return "reject"


def limit_translation_candidates(records, max_per_english=10):
    ranked = sorted(
        records,
        key=lambda row: (
            str(row.get("english_id", "")),
            -float(row.get("confidence", 0.0)),
            -float(row.get("similarity", 0.0)),
        ),
    )
    counts = defaultdict(int)
    result = []
    for record in ranked:
        english_id = str(record.get("english_id", ""))
        if not english_id or counts[english_id] >= max_per_english:
            continue
        counts[english_id] += 1
        result.append(record)
    return result


def build_translation_edges(rows, vectors, english_indices, chinese_indices, threshold, k):
    english_rows = [rows[i] for i in english_indices]
    chinese_rows = [rows[i] for i in chinese_indices]
    english_vectors = vectors[english_indices]
    chinese_vectors = vectors[chinese_indices]
    pairs = list(iter_top_neighbours(english_vectors, chinese_vectors, english_rows, chinese_rows, k))
    edges = []
    review = []

    for english, chinese, distance in pairs:
        similarity = round(1.0 - distance, 4)
        relation = pos_relation(english, chinese)
        confidence = translation_confidence(similarity, english, chinese, False, relation)
        record = {
            "english_id": english["id"], "english_word": english["word"],
            "chinese_id": chinese["id"], "chinese_word": chinese["word"],
            "english_definition": english.get("definition", ""),
            "chinese_definition": chinese.get("definition", ""),
            "similarity": similarity, "confidence": confidence,
            "mutual_neighbour": False, "pos_relation": relation,
        }
        decision = translation_decision(similarity, confidence, english, chinese, relation)
        if decision in {"strong", "probable"}:
            edges.append({**record, "relationship": "translation", "quality": decision})
        elif decision == "review":
            review.append({**record, "relationship": "translation_candidate", "quality": "review"})

    return (
        limit_translation_candidates(edges, max_per_english=10),
        limit_translation_candidates(review, max_per_english=5),
    )


def build_radical_edges(rows, max_group_size=100):
    index = defaultdict(list)
    for row in rows:
        if row["lang"] == "zh":
            for radical in row.get("radicals", []):
                index[radical].append(row)
    edges = []
    for radical, members in index.items():
        if len(members) > max_group_size:
            continue
        for first_index, first in enumerate(members):
            for second in members[first_index + 1:]:
                edges.append({
                    "source": first["id"], "target": second["id"],
                    "source_word": first["word"], "target_word": second["word"],
                    "language": "zh", "relationship": "radical_related",
                    "shared_radicals": [radical], "confidence": 0.25,
                })
    return edges


def build_pinyin_groups(rows):
    exact = defaultdict(list)
    toneless = defaultdict(list)
    for row in rows:
        if row["lang"] != "zh" or not row.get("pinyin"):
            continue
        pinyin = clean(row["pinyin"]).lower()
        exact[pinyin].append(row["id"])
        toneless[remove_tones(pinyin)].append(row["id"])
    return {"exact": dict(exact), "toneless": dict(toneless)}


def build_character_groups(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["lang"] != "zh":
            continue
        for char in row["word"]:
            if has_hanzi(char):
                groups[char].append(row["id"])
    return dict(groups)


def build_character_nodes(rows):
    characters = {}
    for row in rows:
        if row["lang"] != "zh":
            continue
        word = row["word"]
        pinyin_parts = clean(row.get("pinyin", "")).split()
        definition = clean_definition(row.get("definition", ""))
        for position, character in enumerate(word):
            if not has_hanzi(character):
                continue
            node = characters.setdefault(character, {
                "id": f"char_{character}",
                "character": character,
                "readings": defaultdict(lambda: {
                    "meanings": set(), "evidence_words": set(), "levels": set()
                }),
                "source_words": set(),
                "radicals": set(row.get("radicals", [])),
            })
            node["source_words"].add(word)
            node["radicals"].update(row.get("radicals", []))
            pinyin = pinyin_parts[position] if position < len(pinyin_parts) else ""
            reading = node["readings"][pinyin or "unknown"]
            if definition:
                reading["meanings"].add(definition)
            reading["evidence_words"].add(word)
            if row.get("level"):
                reading["levels"].add(row["level"])

    output = {}
    for character, node in sorted(characters.items()):
        output[character] = {
            "id": node["id"],
            "character": character,
            "readings": [
                {
                    "pinyin": pinyin if pinyin != "unknown" else "",
                    "meanings": sorted(reading["meanings"]),
                    "meaning_evidence": "word_derived",
                    "evidence_words": sorted(reading["evidence_words"]),
                    "levels": sorted(reading["levels"]),
                }
                for pinyin, reading in sorted(node["readings"].items())
            ],
            "source_words": sorted(node["source_words"]),
            "radicals": sorted(node["radicals"]),
        }
    return output


def build_character_edges(rows, character_nodes):
    edges = []
    for row in rows:
        if row["lang"] != "zh":
            continue
        pinyin_parts = clean(row.get("pinyin", "")).split()
        for position, character in enumerate(row["word"]):
            if character not in character_nodes:
                continue
            edges.append({
                "source": row["id"], "source_word": row["word"],
                "target": character_nodes[character]["id"],
                "character": character, "position": position,
                "pinyin": pinyin_parts[position] if position < len(pinyin_parts) else "",
                "relationship": "contains_character",
            })
    return edges


def build_meaning_groups(rows, semantic_edges, language, prefix, threshold):
    by_id = {row["id"]: row for row in rows if row["lang"] == language}
    adjacency = defaultdict(set)
    for edge in semantic_edges:
        if edge["language"] != language or edge["similarity"] < threshold:
            continue
        if edge["pos_relation"] == "conflict":
            continue
        adjacency[edge["source"]].add(edge["target"])
        adjacency[edge["target"]].add(edge["source"])

    groups = {}
    visited = set()
    number = 1
    for node_id in by_id:
        if node_id in visited or node_id not in adjacency:
            continue
        stack = [node_id]
        component = []
        visited.add(node_id)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbour in adjacency[current]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    stack.append(neighbour)
        if len(component) < 2:
            continue
        group_id = f"{prefix}_{number:05}"
        groups[group_id] = {
            "id": group_id, "language": language, "type": "meaning_group",
            "members": [by_id[item] for item in sorted(component)],
        }
        number += 1
    return groups


def build_english_domains(rows):
    domain_words = {
        "education": ["school", "teacher", "student", "learn", "study"],
        "emotion": ["happy", "sad", "angry", "fear", "love", "hate"],
        "family": ["family", "father", "mother", "child", "parent"],
        "food": ["food", "eat", "drink", "rice", "water"],
        "money": ["money", "pay", "cost", "price", "buy", "sell"],
        "travel": ["travel", "visit", "arrive", "leave", "journey"],
    }
    ids = {row["word"].lower(): row["id"] for row in rows if row["lang"] == "en"}
    return {
        domain: [ids[word] for word in words if word in ids]
        for domain, words in domain_words.items()
    }


def resolve_pdf(requested: str, label: str) -> Path:
    path = Path(requested).expanduser()
    candidates = [path] if path.is_absolute() else [
        Path.cwd() / path, Path(__file__).resolve().parent / path,
        Path.home() / "Downloads" / path.name, Path.home() / "Desktop" / path.name,
        Path.home() / "Documents" / path.name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"{label} PDF not found: {requested}\nChecked:\n" + "\n".join(map(str, candidates))
    )


def main():
    parser = argparse.ArgumentParser(description="Build complete English and Mandarin vocabulary layers.")
    parser.add_argument("--english", default="ENGLISH.pdf")
    parser.add_argument("--hsk", default="HSK.pdf")
    parser.add_argument("--out", default="output")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--semantic-threshold", type=float, default=0.80)
    parser.add_argument("--meaning-group-threshold", type=float, default=0.90)
    parser.add_argument("--translation-threshold", type=float, default=0.80)
    parser.add_argument("--same-language-k", type=int, default=30)
    parser.add_argument("--translation-k", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--strict-hsk", action="store_true")
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args()

    start = time.perf_counter()
    log("=" * 70)
    log("VOCABULARY BUILDER — COMPLETE BILINGUAL LAYERED GRAPH")
    log("=" * 70)

    english_path = resolve_pdf(args.english, "English")
    hsk_path = resolve_pdf(args.hsk, "HSK")
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)

    log("\n[1/9] Parsing Oxford 5000...")
    english = parse_english(english_path)
    log(f"Extracted: {len(english):,}")
    validate_english(english)

    log("\n[2/9] Parsing HSK 1–9...")
    chinese = parse_hsk(hsk_path, diagnostic=args.diagnostic)
    log(f"Extracted: {len(chinese):,}")
    validate_hsk(chinese, strict=args.strict_hsk)

    rows = english + chinese
    log("\n[3/9] Adding multi-sense WordNet definitions...")
    rows = add_wordnet_definitions(rows)

    log("\n[4/9] Preparing embeddings and radicals...")
    texts = prepare_embeddings(rows)

    from sentence_transformers import SentenceTransformer
    log(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model)
    log(f"Encoding {len(texts):,} nodes...")
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True, batch_size=args.batch_size)

    english_indices = [i for i, row in enumerate(rows) if row["lang"] == "en"]
    chinese_indices = [i for i, row in enumerate(rows) if row["lang"] == "zh"]

    log("\n[5/9] Building English and Mandarin semantic layers...")
    semantic_edges, morphology_edges = build_same_language_edges(
        rows, vectors, {"en": english_indices, "zh": chinese_indices},
        args.semantic_threshold, args.same_language_k,
    )

    log("[6/9] Building translation candidates...")
    translation_edges, translation_review = build_translation_edges(
        rows, vectors, english_indices, chinese_indices,
        args.translation_threshold, args.translation_k,
    )

    log("[7/9] Building Mandarin character and structural layers...")
    radical_edges = build_radical_edges(rows)
    pinyin_groups = build_pinyin_groups(rows)
    character_groups = build_character_groups(rows)
    character_nodes = build_character_nodes(rows)
    character_edges = build_character_edges(rows, character_nodes)

    log("[8/9] Building meaning groups and English domains...")
    english_meaning_groups = build_meaning_groups(
        rows, semantic_edges, "en", "english_meaning", args.meaning_group_threshold,
    )
    mandarin_meaning_groups = build_meaning_groups(
        rows, semantic_edges, "zh", "mandarin_meaning", args.meaning_group_threshold,
    )
    english_domains = build_english_domains(rows)

    nodes = build_nodes(rows)
    (output / "nodes.json").write_text(json.dumps(nodes, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "english_nodes.json").write_text(json.dumps([n for n in nodes if n["lang"] == "en"], ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "mandarin_word_nodes.json").write_text(json.dumps([n for n in nodes if n["lang"] == "zh"], ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "mandarin_character_nodes.json").write_text(json.dumps(character_nodes, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "mandarin_character_edges.json").write_text(json.dumps(character_edges, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "edges_semantic_english.json").write_text(json.dumps([e for e in semantic_edges if e["language"] == "en"], ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "edges_semantic_mandarin.json").write_text(json.dumps([e for e in semantic_edges if e["language"] == "zh"], ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "edges_morphological_english.json").write_text(json.dumps(morphology_edges, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "edges_translation.json").write_text(json.dumps(translation_edges, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "edges_radical.json").write_text(json.dumps(radical_edges, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "english_meaning_groups.json").write_text(json.dumps(english_meaning_groups, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "mandarin_meaning_groups.json").write_text(json.dumps(mandarin_meaning_groups, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "mandarin_pinyin_groups.json").write_text(json.dumps(pinyin_groups, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "mandarin_character_groups.json").write_text(json.dumps(character_groups, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "english_domains.json").write_text(json.dumps(english_domains, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(nodes).to_csv(output / "entries.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(translation_edges).to_csv(output / "translations.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(translation_review).to_csv(output / "translation_review.csv", index=False, encoding="utf-8-sig")

    diagnostics = {
        "english_nodes": len(english),
        "mandarin_word_nodes": len(chinese),
        "total_word_nodes": len(nodes),
        "mandarin_character_nodes": len(character_nodes),
        "mandarin_character_edges": len(character_edges),
        "english_semantic_edges": sum(e["language"] == "en" for e in semantic_edges),
        "mandarin_semantic_edges": sum(e["language"] == "zh" for e in semantic_edges),
        "english_morphological_edges": len(morphology_edges),
        "translation_edges": len(translation_edges),
        "translation_review_candidates": len(translation_review),
        "radical_edges": len(radical_edges),
        "english_meaning_groups": len(english_meaning_groups),
        "mandarin_meaning_groups": len(mandarin_meaning_groups),
        "exact_pinyin_groups": len(pinyin_groups["exact"]),
        "toneless_pinyin_groups": len(pinyin_groups["toneless"]),
        "character_groups": len(character_groups),
        "english_domains": len(english_domains),
        "model": args.model,
        "semantic_threshold": args.semantic_threshold,
        "meaning_group_threshold": args.meaning_group_threshold,
        "translation_threshold": args.translation_threshold,
        "same_language_k": args.same_language_k,
        "translation_k": args.translation_k,
        "elapsed_seconds": round(time.perf_counter() - start, 2),
    }
    (output / "diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")

    log("\n[9/9] SUCCESS")
    log(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    log(f"Output: {output}")


if __name__ == "__main__":
    main()
