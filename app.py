from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import unicodedata
import warnings
from pathlib import Path

import pandas as pd
import pdfplumber


# ============================================================
# Vocabulary Node Builder
# Corrected PDF extraction for:
#   - Oxford 5000 by CEFR level
#   - New HSK Vocabulary Levels 1-6 and 7-9
#
# IMPORTANT:
# The Oxford PDF is multi-column. pdfplumber's normal extract_text()
# flattens columns together, so this version extracts words from the
# individual PDF words/coordinates instead.
#
# The HSK PDF contains some CJK Compatibility / Kangxi characters.
# NFKC normalization converts them back to normal Chinese characters.
# ============================================================


HANZI_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
OXFORD_LEVEL_RE = re.compile(r"^(A1|A2|B1|B2|C1)$", re.I)

OXFORD_POS_RE = re.compile(
    r"^(?P<pos>"
    r"(?:n\.|v\.|adj\.|adv\.|pron\.|prep\.|conj\.|det\.|"
    r"exclam\.|modal v\.|auxiliary|article|number|infinitive)"
    r"(?:\s*,\s*(?:n\.|v\.|adj\.|adv\.|pron\.|prep\.|conj\.|det\.|"
    r"exclam\.|modal v\.|auxiliary|article|number|infinitive))*"
    r")",
    re.I,
)

# HSK entries have:
# number + Chinese + pinyin + POS + definition
# POS/definition may be absent for some Level 7-9 entries.
HSK_ENTRY_RE = re.compile(
    r"^(?P<number>\d+)\s+"
    r"(?P<word>\S+)\s+"
    r"(?P<pinyin>\S+)"
    r"(?:\s+(?P<rest>.*))?$"
)

HSK_LEVEL_RE = re.compile(
    r"(?:NEW\s+HSK\s+VOCABULARY\s*)?"
    r"LEVEL\s*([1-9](?:\s*-\s*[1-9])?)",
    re.I,
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
    "能愿": "modal",
}


# ============================================================
# General helpers
# ============================================================

def clean(text: str | None) -> str:
    if text is None:
        return ""

    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00ad", "")
    text = text.replace("\ufeff", "")
    text = text.replace("\ufffd", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_unicode(text: str) -> str:
    """NFKC is essential for the HSK PDF's compatibility characters."""
    return unicodedata.normalize("NFKC", text or "")


def has_hanzi(text: str) -> bool:
    text = normalize_unicode(text)
    return bool(HANZI_RE.search(text))


def suppress_pdf_font_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"Could not get FontBBox from font descriptor.*",
    )
    logging.getLogger("pdfminer").setLevel(logging.ERROR)


def open_pdf(path: Path):
    suppress_pdf_font_warnings()
    return pdfplumber.open(path)


# ============================================================
# Oxford 5000
# ============================================================

def normalize_oxford_pos(text: str) -> str:
    text = clean(text)

    mapping = {
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
        "number": "number",
        "article": "article",
        "auxiliary": "auxiliary",
        "infinitive": "infinitive",
    }

    parts = []
    for piece in re.split(r"\s*,\s*", text):
        piece = clean(piece)
        if piece in mapping and mapping[piece] not in parts:
            parts.append(mapping[piece])

    return ", ".join(parts)


def looks_like_oxford_word(text: str) -> bool:
    text = clean(text)

    if not text:
        return False

    # Vocabulary may contain spaces, hyphens, apostrophes and numbers.
    return bool(
        re.fullmatch(
            r"[A-Za-z][A-Za-z0-9'’\-]*(?:\s+[A-Za-z0-9][A-Za-z0-9'’\-]*)*",
            text,
        )
    )


def parse_oxford_cell(cell_text: str, level: str):
    """
    Parse one Oxford vocabulary entry.

    Oxford uses several POS formats, including:

        n.
        v.
        adj.
        adv.
        det./adj.
        det./pron.
        exclam./n.
        modal v.
        auxiliary v.
        article
        number
        infinitive

    Some entries also contain a parenthetical clarification instead of
    an explicit POS marker, e.g.:

        match (contest/correspond)

    Those are retained with an empty POS rather than being discarded.
    """

    text = clean(cell_text)

    if not text:
        return None

    text = re.sub(r"^[•·]+", "", text).strip()

    # --------------------------------------------------------
    # POS patterns
    # --------------------------------------------------------

    pos_pattern = re.compile(
        r"(?:"
        r"modal\s+v\."
        r"|auxiliary\s+v\."
        r"|infinitive"
        r"|article"
        r"|number"
        r"|n\."
        r"|v\."
        r"|adj\."
        r"|adv\."
        r"|pron\."
        r"|prep\."
        r"|conj\."
        r"|det\."
        r"|exclam\."
        r")",
        re.I,
    )

    match = pos_pattern.search(text)

    # --------------------------------------------------------
    # Normal POS entry
    # --------------------------------------------------------

    if match:

        word = clean(text[:match.start()])
        pos_text = clean(text[match.start():])

        # Remove common Oxford punctuation between POS labels.
        pos_text = pos_text.replace(";", ",")
        pos_text = re.sub(r"/", ",", pos_text)

        if not looks_like_oxford_word(word):
            return None

        pos = normalize_oxford_pos_flexible(pos_text)

        return {
            "id": "",
            "lang": "en",
            "word": word.lower(),
            "pinyin": "",
            "pos": pos,
            "definition": "",
            "level": level,
            "source": "",
        }

    # --------------------------------------------------------
    # Entries without explicit POS
    #
    # Example:
    #     match (contest/correspond)
    #
    # Retain them rather than throwing them away.
    # --------------------------------------------------------

    word = text

    # Remove obvious page/header material.
    if word.lower() in {
        "level",
        "by cefr",
        "words to learn in english",
        "from a1 to b2 level.",
    }:
        return None

    # Remove parenthetical sense information from the vocabulary
    # word only when the whole entry clearly follows:
    #
    #     word (clarification)
    #
    # We retain the clarification as the definition.
    definition = ""

    parenthetical = re.match(
        r"^(?P<word>.+?)\s+\((?P<definition>.+)\)$",
        text,
    )

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

def normalize_oxford_pos_flexible(text: str) -> str:
    text = clean(text)

    mapping = {
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

    found = []

    # Normalize separators used by Oxford.
    text = text.replace(";", ",")
    text = re.sub(r"/", ",", text)

    # Match longest forms first.
    patterns = sorted(
        mapping.keys(),
        key=len,
        reverse=True,
    )

    for pattern in patterns:

        if re.search(
            rf"(?<![A-Za-z]){re.escape(pattern)}(?![A-Za-z])",
            text,
            re.I,
        ):
            value = mapping[pattern]

            if value not in found:
                found.append(value)

    return ", ".join(found)
    
def get_page_columns(page):
    """
    Recover the ACTUAL four visual vocabulary columns used by the
    Oxford PDF.

    The PDF is physically arranged as four vocabulary columns across
    the page, but pdfplumber's word extraction can make the two
    side-by-side vocabulary streams appear merged.

    We therefore divide the page using its horizontal midpoint first,
    then divide each half vertically into its two actual columns.
    """

    words = page.extract_words(
        x_tolerance=1.5,
        y_tolerance=2,
        keep_blank_chars=False,
    )

    if not words:
        return []

    page_width = float(page.width)
    midpoint = page_width / 2

    left = []
    right = []

    for word in words:
        x0 = float(word["x0"])
        x1 = float(word["x1"])
        centre = (x0 + x1) / 2

        if centre < midpoint:
            left.append(word)
        else:
            right.append(word)

    # Sort each half by x-coordinate.
    left.sort(key=lambda w: float(w["x0"]))
    right.sort(key=lambda w: float(w["x0"]))

    def split_half(words):
        if not words:
            return []

        xs = sorted(
            [
                (
                    (float(w["x0"]) + float(w["x1"])) / 2,
                    index,
                    w,
                )
                for index, w in enumerate(words)
            ],
            key=lambda item: item[0],
        )

        # Find the largest horizontal gap inside this half.
        gaps = []

        for i in range(1, len(xs)):
            gap = xs[i][0] - xs[i - 1][0]
            gaps.append((gap, i))

        if not gaps:
            return [words]

        largest_gap, split_index = max(
            gaps,
            key=lambda item: item[0],
        )

        # If there isn't a meaningful gap, keep the half intact.
        if largest_gap < page_width * 0.02:
            return [words]

        first = [
            item[2]
            for item in xs[:split_index]
        ]

        second = [
            item[2]
            for item in xs[split_index:]
        ]

        return [first, second]

    columns = split_half(left) + split_half(right)

    return [
        column
        for column in columns
        if len(column) >= 2
    ]


def column_lines(words):
    """Turn coordinate-positioned PDF words into visual lines."""
    if not words:
        return []

    words = sorted(
        words,
        key=lambda w: (
            round(float(w["top"]), 1),
            float(w["x0"]),
        ),
    )

    lines = []

    for word in words:
        top = float(word["top"])

        target = None

        for line in reversed(lines[-3:]):
            avg_top = sum(x["_top"] for x in line) / len(line)
            if abs(top - avg_top) <= 3.0:
                target = line
                break

        if target is None:
            lines.append([{
                **word,
                "_top": top,
            }])
        else:
            target.append({
                **word,
                "_top": top,
            })

    result = []

    for line in lines:
        line.sort(key=lambda w: float(w["x0"]))
        text = clean(" ".join(w["text"] for w in line))

        if text:
            result.append(text)

    return result


def parse_english(path: Path) -> list[dict]:
    """
    Extract Oxford 5000 from the actual multi-column PDF.

    CEFR headings are carried forward. Individual columns are processed
    separately so entries from adjacent columns cannot merge together.
    """
    results = []
    current_level = ""

    with open_pdf(path) as pdf:
        for page_number, page in enumerate(pdf.pages, 1):

            # First look for CEFR heading in the page text.
            page_text = clean(page.extract_text() or "")

            for possible in re.findall(
                r"\b(?:A1|A2|B1|B2|C1)\b",
                page_text,
                re.I,
            ):
                current_level = possible.upper()

            columns = get_page_columns(page)

            for column in columns:
                lines = column_lines(column)

                if page_number >= 10:
                    for line in lines:
                        if line in {"A1", "A2", "B1", "B2", "C1"}:
                            print(
                                f"\nFOUND CEFR {line} "
                                f"ON PAGE {page_number}"
                            )
            
                for line in lines:
                    level_match = OXFORD_LEVEL_RE.fullmatch(clean(line))
                    if level_match:
                        current_level = level_match.group(1).upper()
                        continue

                    if not current_level:
                        continue

                    entry = parse_oxford_cell(line, current_level)

                    if entry:
                        entry["_page"] = page_number
                        results.append(entry)

    # Remove duplicates while preserving first occurrence.
    unique = {}

    for row in results:
        key = row["word"]

        if key not in unique:
            unique[key] = row

    results = list(unique.values())

    for i, row in enumerate(results, 1):
        row["id"] = f"en_{i:05}"
        row["source"] = path.name
        row.pop("_page", None)

    return results


# ============================================================
# HSK 1-9
# ============================================================

def normalize_zh_pos(text: str) -> str:
    text = clean(text)

    found = []

    # Longest marker first.
    for key in sorted(ZH_POS, key=len, reverse=True):
        if key in text and ZH_POS[key] not in found:
            found.append(ZH_POS[key])

    return ", ".join(found)


def parse_hsk_entry_line(line: str, level: str):
    line = clean(line)

    match = HSK_ENTRY_RE.match(line)

    if not match:
        return None

    number = match.group("number")
    word = normalize_unicode(clean(match.group("word")))
    pinyin = clean(match.group("pinyin"))
    rest = clean(match.group("rest"))

    if not has_hanzi(word):
        return None

    pos = ""
    definition = ""

    if rest:
        # POS usually appears immediately after pinyin.
        pos_match = re.match(
            r"^(?P<pos>"
            r"(?:名|动|形|副|代|介|连|助|量|数|能愿)"
            r"(?:\s*[、,]\s*(?:名|动|形|副|代|介|连|助|量|数|能愿))*"
            r"|(?:noun|verb|adjective|adverb|pronoun|preposition|"
            r"conjunction|particle|classifier|number)"
            r"(?:\s*[,、]\s*(?:noun|verb|adjective|adverb|pronoun|"
            r"preposition|conjunction|particle|classifier|number))*"
            r")\s*",
            rest,
            re.I,
        )

        if pos_match:
            pos = normalize_zh_pos(pos_match.group("pos"))
            if not pos:
                pos = clean(pos_match.group("pos"))
            definition = clean(rest[pos_match.end():])
        else:
            # Level 7-9 often has no translation/POS in this PDF.
            definition = rest

    return {
        "id": "",
        "lang": "zh",
        "word": word,
        "pinyin": pinyin,
        "pos": pos,
        "definition": definition,
        "level": level,
        "source": "",
        "_number": int(number),
    }


def detect_hsk_level(text: str):
    text = normalize_unicode(text)

    match = HSK_LEVEL_RE.search(text)

    if not match:
        return None

    raw = re.sub(r"\s+", "", match.group(1))

    if "-" in raw:
        parts = raw.split("-")
        return f"HSK {parts[0]}-{parts[-1]}"

    return f"HSK {raw}"


def parse_hsk(path: Path) -> list[dict]:
    """
    Parse all HSK sections in the supplied 11,000-word PDF.

    The parser keeps each section separate so we can validate against:
        HSK1   300
        HSK2   200
        HSK3   500
        HSK4  1000
        HSK5  1600
        HSK6  1800
        HSK7-9 5600
    """
    
        # --------------------------------------------------------
    # TEMPORARY HSK PDF DIAGNOSTIC
    # --------------------------------------------------------

    with open_pdf(path) as pdf:
        print("\n" + "=" * 70)
        print("HSK PDF DIAGNOSTIC — FIRST 5 PAGES")
        print("=" * 70)

        for page_number, page in enumerate(pdf.pages[:5], 1):
            print(f"\n--- PAGE {page_number} ---")

            text = page.extract_text() or ""

            print(text[:5000])

        print("\n" + "=" * 70)
        print("END HSK DIAGNOSTIC")
        print("=" * 70)
        
    results = []
    current_level = None
    pending = None

    counts = {}

    with open_pdf(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""

            for raw_line in text.splitlines():
                line = clean(raw_line)

                if not line:
                    continue

                level = detect_hsk_level(line)

                if level:
                    # Finish previous wrapped entry.
                    if pending:
                        results.append(pending)
                        pending = None

                    current_level = level
                    counts.setdefault(current_level, 0)
                    continue

                if current_level is None:
                    continue

                upper = line.upper()

                if (
                    "NO. WORD PINYIN" in upper
                    or upper == "ENTRIES"
                    or "MANDARINBEAN.COM PAGE" in upper
                    or line.startswith("⇨")
                    or line.startswith(">>>")
                ):
                    continue

                parsed = parse_hsk_entry_line(line, current_level)

                if parsed:
                    if pending:
                        results.append(pending)

                    pending = parsed
                    continue

                # A line that is not a new numbered entry can be:
                #   - continuation of the translation
                #   - continuation of POS
                #   - a wrapped Level 7-9 entry
                if pending:
                    continuation = normalize_unicode(line)

                    if not re.fullmatch(r"\d+", continuation):
                        pending["definition"] = clean(
                            f'{pending["definition"]} {continuation}'
                        )

    if pending:
        results.append(pending)

    # Deduplicate by Chinese word, but preserve the first HSK level.
    unique = {}

    for row in results:
        row.pop("_number", None)

        key = row["word"]

        if key not in unique:
            unique[key] = row
        else:
            # If the duplicate has a richer definition, retain it.
            if len(row["definition"]) > len(unique[key]["definition"]):
                unique[key]["definition"] = row["definition"]

    results = list(unique.values())

    for i, row in enumerate(results, 1):
        row["id"] = f"zh_{i:05}"
        row["source"] = path.name

    return results


# ============================================================
# Validation
# ============================================================

def validate_english(rows: list[dict]) -> None:
    print("\nOxford CEFR extraction check:")

    counts = {}

    for row in rows:
        counts[row["level"]] = counts.get(row["level"], 0) + 1

    for level in ["A1", "A2", "B1", "B2", "C1"]:
        print(f"  {level}: {counts.get(level, 0):,}")

    print(f"  TOTAL: {len(rows):,}")

    if len(rows) < 4000:
        raise RuntimeError(
            f"Oxford extraction is still too low: {len(rows):,}. "
            "The program will NOT continue to semantic embedding."
        )

    if len(rows) > 5500:
        print(
            "Warning: Oxford extraction is above 5,500. "
            "Check for duplicate/header entries."
        )


def validate_hsk(rows: list[dict]) -> None:
    counts = {}

    for row in rows:
        level = row["level"]
        counts[level] = counts.get(level, 0) + 1

    print("\nHSK extraction check:")

    for level, expected in HSK_EXPECTED.items():
        actual = counts.get(level, 0)
        print(
            f"  {level:<8} {actual:>6,} / expected {expected:,}"
        )

    print(f"  TOTAL:   {len(rows):,} / expected 11,000")

    # We need essentially the whole vocabulary before building the graph.
    if len(rows) < 10000:
        raise RuntimeError(
            f"HSK extraction is still too low: {len(rows):,} / 11,000. "
            "The program will NOT continue to semantic embedding."
        )


# ============================================================
# WordNet
# ============================================================

def add_wordnet_definitions(rows: list[dict]) -> list[dict]:
    try:
        import nltk
        from nltk.corpus import wordnet as wn
    except ImportError:
        print(
            "Warning: NLTK unavailable; WordNet skipped.",
            file=sys.stderr,
        )
        return rows

    try:
        wn.synsets("test")
    except LookupError:
        try:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
        except Exception as exc:
            print(
                f"Warning: WordNet download failed: {exc}",
                file=sys.stderr,
            )
            return rows

    for row in rows:
        if row["lang"] != "en" or row["definition"]:
            continue

        try:
            synsets = wn.synsets(row["word"])
        except Exception:
            synsets = []

        if synsets:
            row["definition"] = synsets[0].definition()

    return rows


# ============================================================
# Radicals
# ============================================================

def radicals(word: str) -> list[str]:
    word = normalize_unicode(word)

    try:
        from cjkradlib import RadicalFinder

        finder = RadicalFinder(lang="zh")
        found = []

        for char in word:
            if not has_hanzi(char):
                continue

            try:
                result = finder.search(char)
                values = list(
                    getattr(result, "compositions", []) or []
                )
                found.extend(values if values else [char])
            except Exception:
                found.append(char)

        return sorted(set(found))

    except Exception:
        return sorted(
            set(char for char in word if has_hanzi(char))
        )


# ============================================================
# Semantic graph
# ============================================================

def make_graph(rows: list[dict], threshold: float = 0.78):
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.neighbors import NearestNeighbors
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependencies. Install:\n"
            "pip install sentence-transformers scikit-learn"
        ) from exc

    for row in rows:
        row["radicals"] = (
            radicals(row["word"])
            if row["lang"] == "zh"
            else []
        )

        pieces = [
            row["word"],
            row["definition"],
            row["pos"],
            row["pinyin"] if row["lang"] == "zh" else "",
        ]

        row["embedding_text"] = " | ".join(
            clean(x) for x in pieces if clean(x)
        )

    print("\nLoading multilingual embedding model...")
    model = SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2"
    )

    print(f"Encoding {len(rows):,} vocabulary entries...")

    vectors = model.encode(
        [row["embedding_text"] for row in rows],
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
    )

    k = min(30, len(rows))

    if k < 2:
        concepts = {
            "concept_00001": {
                "id": "concept_00001",
                "definition": rows[0]["definition"],
                "members": [rows[0]],
            }
        }
        return concepts, {}, []

    nn = NearestNeighbors(
        n_neighbors=k,
        metric="cosine",
        n_jobs=-1,
    ).fit(vectors)

    distances, indexes = nn.kneighbors(vectors)

    parent = list(range(len(rows)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    review = []

    for i in range(len(rows)):
        for distance, j in zip(
            distances[i][1:],
            indexes[i][1:],
        ):
            score = 1.0 - float(distance)

            if score >= threshold:
                union(i, int(j))

            elif (
                score >= threshold - 0.06
                and rows[i]["lang"] != rows[int(j)]["lang"]
            ):
                review.append({
                    "entry_a": rows[i]["id"],
                    "word_a": rows[i]["word"],
                    "entry_b": rows[int(j)]["id"],
                    "word_b": rows[int(j)]["word"],
                    "similarity": round(score, 4),
                })

    groups = {}

    for i, row in enumerate(rows):
        groups.setdefault(find(i), []).append(row)

    concepts = {}
    radical_index = {}

    for number, members in enumerate(groups.values(), 1):
        concept_id = f"concept_{number:05}"

        label = max(
            members,
            key=lambda x: len(x["definition"]),
        )

        concepts[concept_id] = {
            "id": concept_id,
            "definition": label["definition"],
            "members": [
                {
                    key: member[key]
                    for key in [
                        "id",
                        "lang",
                        "word",
                        "pinyin",
                        "pos",
                        "definition",
                        "level",
                        "radicals",
                    ]
                }
                for member in members
            ],
        }

        for member in members:
            for radical in member["radicals"]:
                radical_index.setdefault(
                    radical,
                    [],
                ).append(concept_id)

    return concepts, radical_index, review


# ============================================================
# File resolution
# ============================================================

def resolve_pdf(requested: str, label: str) -> Path:
    path = Path(requested).expanduser()

    candidates = []

    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([
            Path.cwd() / path,
            Path(__file__).resolve().parent / path,
            Path.home() / "Downloads" / path.name,
            Path.home() / "Desktop" / path.name,
            Path.home() / "Documents" / path.name,
        ])

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    checked = "\n".join(
        f"  - {candidate}"
        for candidate in candidates
    )

    raise FileNotFoundError(
        f"{label} PDF not found.\n\n"
        f"Requested: {requested}\n\n"
        f"Checked:\n{checked}"
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build a bilingual English-Chinese vocabulary "
            "semantic concept graph."
        )
    )

    parser.add_argument(
        "--english",
        default="ENGLISH.pdf",
    )

    parser.add_argument(
        "--hsk",
        default="HSK.pdf",
    )

    parser.add_argument(
        "--out",
        default="output",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.78,
    )

    args = parser.parse_args()

    print("=" * 70)
    print("VOCABULARY NODE BUILDER — CORRECTED PDF EXTRACTION")
    print("=" * 70)

    english_path = resolve_pdf(
        args.english,
        "English",
    )

    hsk_path = resolve_pdf(
        args.hsk,
        "HSK",
    )

    print(f"\nEnglish PDF: {english_path}")
    print(f"HSK PDF:     {hsk_path}")

    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Extraction
    # --------------------------------------------------------

    print("\n[1/5] Parsing Oxford 5000...")
    english = parse_english(english_path)

    print(
        f"      Extracted: {len(english):,}"
    )

    validate_english(english)

    print("\n[2/5] Parsing HSK 1–9...")
    chinese = parse_hsk(hsk_path)

    print(
        f"      Extracted: {len(chinese):,}"
    )

    validate_hsk(chinese)

    # --------------------------------------------------------
    # Definitions
    # --------------------------------------------------------

    print("\n[3/5] Adding English WordNet definitions...")
    english = add_wordnet_definitions(english)

    rows = english + chinese

    print(
        f"\nTOTAL VOCABULARY: {len(rows):,}"
    )

    # --------------------------------------------------------
    # Semantic graph
    # --------------------------------------------------------

    print("\n[4/5] Building semantic graph...")

    concepts, radical_index, review = make_graph(
        rows,
        args.threshold,
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print("\n[5/5] Writing output...")

    pd.DataFrame(rows).to_csv(
        output / "entries.csv",
        index=False,
        encoding="utf-8-sig",
    )

    (output / "concepts.json").write_text(
        json.dumps(
            concepts,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    (output / "radicals.json").write_text(
        json.dumps(
            radical_index,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    pd.DataFrame(review).drop_duplicates().to_csv(
        output / "review.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 70)
    print("SUCCESS")
    print("=" * 70)
    print(f"Oxford 5000 : {len(english):,}")
    print(f"HSK 1–9     : {len(chinese):,}")
    print(f"TOTAL       : {len(rows):,}")
    print(f"CONCEPTS    : {len(concepts):,}")
    print(f"OUTPUT      : {output}")
    print("=" * 70)


if __name__ == "__main__":
    main()