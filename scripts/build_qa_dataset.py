"""Check the hand-written question-answer file and split it into train / val / test.

Input:  data/qa/verdict_qa.jsonl   one question per line (see the README section below)
Output: data/qa/train.jsonl, data/qa/val.jsonl, data/qa/test.jsonl

Each line of verdict_qa.jsonl is:
    {"fact": "F12", "type": "reason", "question": "...", "answer": "...", "evidence": ["exact quote", ...]}
  fact      - group id. All questions about the same fact share it and always land in the
              same split, so a reworded test question never has a twin in training.
              Ids starting with "U" are questions the story cannot answer.
  evidence  - exact quotes from the story that support the answer (empty for "U" facts).

Checks (the script stops with a list of problems if any fail):
  - every line is valid JSON with the required fields and a known type
  - every evidence quote appears word-for-word in the story
  - unanswerable questions have an answer starting "The story does not say" and no evidence
  - no question appears twice
  - every fact statement (fact_statements.jsonl) belongs to a known story fact and is unique
Warnings (printed, not fatal): questions in different splits that share most of their words,
and training questions that look like one of the held-out probe questions (probe.jsonl).

Splits are frozen in data/qa/fact_splits.json: once a fact is in test it stays in test, even
when questions are added. A brand-new fact must be added to that file by hand.

Also writes data/qa/review_statements.md: each fact's evidence quotes next to its rewritten
statements, for checking by eye.

Usage:
    python scripts/build_qa_dataset.py
"""
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STORY = ROOT / "data" / "raw" / "verdict" / "the-verdict.txt"
QA_DIR = ROOT / "data" / "qa"
SOURCE = QA_DIR / "verdict_qa.jsonl"
STATEMENTS = QA_DIR / "fact_statements.jsonl"
PROBE = QA_DIR / "probe.jsonl"
SPLITS_FILE = QA_DIR / "fact_splits.json"

TYPES = {"character", "relationship", "event", "detail", "reason", "multi-hop", "interpretation", "unanswerable"}
UNANSWERABLE = "The story does not say"  # every unanswerable answer starts with this
SEED = 42
CHUNK = 7  # story facts are split in runs of 7: 1 test, 1 val, 5 train


def squash(text: str) -> str:
    """Collapse all whitespace so line breaks in the story don't break quote matching."""
    return re.sub(r"\s+", " ", text).strip()


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z']+", text.lower()))


def load_and_check(story: str) -> list[dict]:
    rows, problems, seen = [], [], {}
    for n, line in enumerate(SOURCE.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            problems.append(f"line {n}: not valid JSON ({e})")
            continue
        missing = {"fact", "type", "question", "answer", "evidence"} - row.keys()
        if missing:
            problems.append(f"line {n}: missing {sorted(missing)}")
            continue
        if row["type"] not in TYPES:
            problems.append(f"line {n}: unknown type {row['type']!r}")
        is_u = row["fact"].startswith("U")
        if is_u != (row["type"] == "unanswerable"):
            problems.append(f"line {n}: fact ids starting with U must have type 'unanswerable' and vice versa")
        if is_u and (row["evidence"] or not row["answer"].startswith(UNANSWERABLE)):
            problems.append(f"line {n}: unanswerable questions need an answer starting {UNANSWERABLE!r} "
                            "and no evidence")
        if not is_u and not row["evidence"]:
            problems.append(f"line {n}: answerable question has no evidence")
        positions = []
        for quote in row["evidence"]:
            pos = story.find(squash(quote))
            if pos < 0:
                problems.append(f"line {n}: evidence not found in the story: {quote[:70]!r}")
            positions.append(pos)
        row["story_pos"] = min(positions) if positions else -1
        if row["question"].lower() in seen:
            problems.append(f"line {n}: duplicate of line {seen[row['question'].lower()]}: {row['question']!r}")
        seen[row["question"].lower()] = n
        row["line"] = n
        rows.append(row)

    if problems:
        print(f"Found {len(problems)} problem(s) in {SOURCE.relative_to(ROOT)}:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    return rows


def frozen_splits(rows: list[dict]) -> dict[str, str]:
    """Use the saved fact -> split assignment; create it on the very first run."""
    if not SPLITS_FILE.exists():
        assign = split_facts(rows)
        SPLITS_FILE.write_text(json.dumps(dict(sorted(assign.items())), indent=1))
        return assign
    assign = json.loads(SPLITS_FILE.read_text())
    unknown = sorted({r["fact"] for r in rows} - assign.keys())
    if unknown:
        print(f"Facts not in {SPLITS_FILE.name}: {', '.join(unknown)}. Add each one to that file with "
              "\"train\", \"val\" or \"test\" before building.")
        sys.exit(1)
    return assign


def load_statements(rows: list[dict]) -> list[dict]:
    """Check the rewritten fact statements and return them."""
    if not STATEMENTS.exists():
        return []
    story_facts = {r["fact"] for r in rows if not r["fact"].startswith("U")}
    statements, problems, seen = [], [], set()
    for n, line in enumerate(STATEMENTS.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            s = json.loads(line)
        except json.JSONDecodeError as e:
            problems.append(f"line {n}: not valid JSON ({e})")
            continue
        if s.get("fact") not in story_facts:
            problems.append(f"line {n}: fact {s.get('fact')!r} is not a story fact in {SOURCE.name}")
        if not s.get("statement", "").strip():
            problems.append(f"line {n}: empty statement")
        elif s["statement"].lower() in seen:
            problems.append(f"line {n}: duplicate statement {s['statement']!r}")
        seen.add(s.get("statement", "").lower())
        statements.append(s)
    if problems:
        print(f"Found {len(problems)} problem(s) in {STATEMENTS.relative_to(ROOT)}:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    return statements


def write_review(rows: list[dict], statements: list[dict]) -> Path:
    """One section per fact: its evidence quotes, then its rewritten statements."""
    evidence, stmts = {}, {}
    for r in rows:
        for q in r["evidence"]:
            evidence.setdefault(r["fact"], [])
            if q not in evidence[r["fact"]]:
                evidence[r["fact"]].append(q)
    for s in statements:
        stmts.setdefault(s["fact"], []).append(s["statement"])
    lines = ["# Fact statements to review", "",
             "For each fact: are all statements supported by the quotes, with nothing added?", ""]
    for fact in sorted(stmts):
        lines.append(f"## {fact}")
        lines += [f"> {q}" for q in evidence.get(fact, [])]
        lines.append("")
        lines += [f"{i}. {s}" for i, s in enumerate(stmts[fact], start=1)]
        lines.append("")
    out = QA_DIR / "review_statements.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def split_facts(rows: list[dict]) -> dict[str, str]:
    """Assign every fact id to train / val / test (only used to create fact_splits.json)."""
    rng = random.Random(SEED)
    first_pos = {}
    for r in rows:
        first_pos[r["fact"]] = min(first_pos.get(r["fact"], 10**9), r["story_pos"] if r["story_pos"] >= 0 else 10**9)
    story_facts = sorted((f for f in first_pos if not f.startswith("U")), key=first_pos.get)
    unanswerable = sorted(f for f in first_pos if f.startswith("U"))

    assign = {}
    # Walk through the story in runs of CHUNK facts, so test and val questions come from
    # every part of the story rather than bunching up at the start or end.
    for i in range(0, len(story_facts), CHUNK):
        chunk = story_facts[i:i + CHUNK]
        rng.shuffle(chunk)
        for j, fact in enumerate(chunk):
            assign[fact] = "test" if j == 0 and len(chunk) >= 3 else "val" if j == 1 and len(chunk) >= 3 else "train"
    rng.shuffle(unanswerable)
    for j, fact in enumerate(unanswerable):
        assign[fact] = "test" if j < 2 else "val" if j < 4 else "train"
    return assign


def near_duplicates(rows: list[dict], assign: dict[str, str]) -> list[tuple]:
    """Pairs of questions in DIFFERENT splits whose words overlap a lot (Jaccard >= 0.6)."""
    stop = {"the", "a", "an", "of", "to", "did", "does", "do", "is", "was", "what", "who", "how", "why",
            "in", "his", "her", "he", "she", "jack", "jack's", "gisburn", "gisburn's"}
    flagged = []
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            if assign[a["fact"]] == assign[b["fact"]]:
                continue
            wa, wb = words(a["question"]) - stop, words(b["question"]) - stop
            if wa and wb and len(wa & wb) / len(wa | wb) >= 0.6:
                flagged.append((a, b))
    return flagged


def main() -> None:
    story = squash(STORY.read_text(encoding="utf-8"))
    rows = load_and_check(story)
    assign = frozen_splits(rows)
    statements = load_statements(rows)

    out = {"train": [], "val": [], "test": []}
    for r in rows:
        split = assign[r["fact"]]
        out[split].append({k: r[k] for k in ("fact", "type", "question", "answer", "evidence")} | {"split": split})
    for split, items in out.items():
        with open(QA_DIR / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    n_facts = Counter(assign.values())
    print(f"All checks passed: {len(rows)} questions about {len(assign)} facts; "
          f"every evidence quote was found in the story.\n")
    print(f"{'split':6} {'facts':>6} {'questions':>10}   question types")
    for split in ("train", "val", "test"):
        types = Counter(r["type"] for r in out[split])
        print(f"{split:6} {n_facts[split]:>6} {len(out[split]):>10}   "
              + ", ".join(f"{t} {c}" for t, c in sorted(types.items())))
    print(f"\nTest facts: {', '.join(sorted(f for f, s in assign.items() if s == 'test'))}")
    print(f"Val facts:  {', '.join(sorted(f for f, s in assign.items() if s == 'val'))}")

    flagged = near_duplicates(rows, assign)
    if flagged:
        print(f"\nWarning: {len(flagged)} question pair(s) in different splits look similar; check they are "
              "about different facts:")
        for a, b in flagged:
            print(f"  [{assign[a['fact']]}] {a['question']}\n  [{assign[b['fact']]}] {b['question']}\n")
    else:
        print("\nNo similar-looking questions across splits.")

    if PROBE.exists():
        probe = [json.loads(line) for line in PROBE.read_text(encoding="utf-8").splitlines() if line.strip()]
        probe_assign = {**assign, **{f"probe:{p['question']}": "probe" for p in probe}}
        pairs = near_duplicates([{**p, "fact": f"probe:{p['question']}"} for p in probe] + out["train"],
                                probe_assign)
        pairs = [(a, b) for a, b in pairs if "probe" in (probe_assign[a["fact"]], probe_assign[b["fact"]])]
        if pairs:
            print(f"\nWarning: {len(pairs)} training question(s) look like held-out probe questions:")
            for a, b in pairs:
                print(f"  [probe] {a['question']}\n  [train] {b['question']}\n")
        else:
            print(f"No training question looks like any of the {len(probe)} probe questions.")

    if statements:
        per_fact = Counter(s["fact"] for s in statements)
        review = write_review(rows, statements)
        print(f"\nFact statements: {len(statements)} for {len(per_fact)} facts "
              f"({min(per_fact.values())}-{max(per_fact.values())} each). Review page: {review.relative_to(ROOT)}")
    print(f"Wrote {', '.join(str((QA_DIR / f'{s}.jsonl').relative_to(ROOT)) for s in out)}")


if __name__ == "__main__":
    main()
