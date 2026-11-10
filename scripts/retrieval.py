"""Find the passages of "The Verdict" that answer a question, remembering the conversation.

The story is cut into short passages (about 130 tokens, so one passage plus a question and
its answer fits in the model's 256-token context). A question is matched to passages with
BM25, the classic keyword-search score: a passage scores high when it contains the
question's words, rare words (names, "donkey") count more than common ones, and a word
repeated many times stops adding much.

Follow-up questions: "Who is Stroud?" then "How did he die?". The second question alone
has no name in it, so the search query also includes the words of the previous questions.
They get a small weight normally and a large one when the question looks like a follow-up
(it uses he / she / it / they ... or has very few words of its own), so a new topic still
wins over the old one.

Usage:
    python scripts/retrieval.py "Who is Stroud?"          # show the best passages
    python scripts/retrieval.py                            # interactive, with memory
"""
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STORY_PATH = ROOT / "data" / "raw" / "verdict" / "the-verdict.txt"
PASSAGE_TOKENS = 130  # leaves room for "Context/Question/Answer", the question and a 60-token answer

STOPWORDS = set("""
a an the and or but if of to in on at by for with from as into about than then so
is are was were be been being am do does did done have has had having can could will would
shall should may might must i me my you your we our us
he him his himself she her hers herself they them their it its
what who whom whose which when where why how that this these those there here
not no yes very just also too only ever any some such own same other
tell describe according story verdict
""".split())

# Words that point back to something said earlier.
FOLLOWUP_WORDS = {"he", "him", "his", "himself", "she", "her", "hers", "herself", "they", "them",
                  "their", "it", "its", "this", "that", "these", "those", "there", "then"}


# Word forms a suffix rule cannot join ("died" -> "die", "bought" -> "buy").
WORD_FAMILY = {"died": "die", "dies": "die", "dying": "die", "dead": "die", "death": "die",
               "married": "marry", "marries": "marry", "marriage": "marry", "wedding": "marry",
               "bought": "buy", "gave": "give", "given": "give", "made": "make", "said": "say",
               "saw": "see", "seen": "see", "thought": "think", "told": "tell", "wife": "wife",
               "painter": "paint", "painters": "paint", "began": "begin", "begun": "begin"}


def _stem(word: str) -> str:
    """A very small stemmer so "painting", "painted" and "paints" all match "paint", then cut
    to 6 letters so "wealthy" / "wealth" and "fashionable" / "fashion" match too."""
    if word in WORD_FAMILY:
        return WORD_FAMILY[word]
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            word = word[: -len(suffix)]
            break
    return word[:6]


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower().replace("'s", "").replace("’s", ""))


def terms(text: str) -> list[str]:
    """The searchable words of a text: lowercased, stemmed, without filler words."""
    return [_stem(w) for w in words(text) if w not in STOPWORDS and len(w) > 1]


def normalize(text: str) -> str:
    """For checking whether a quote appears in a passage: letters and digits only."""
    return " ".join(words(text))


# ---------------------------------------------------------------------------- passages

def _sentences(paragraph: str) -> list[str]:
    paragraph = " ".join(paragraph.split())
    # split after . ! ? (and a closing quote) before a capital, but not after "Mr." / "Mrs."
    parts = re.split(r'(?:(?<=[.!?])|(?<=[.!?]["”]))(?<!\bMr\.)(?<!\bMrs\.)(?<!\bDr\.)(?<!\bSt\.)\s+(?=["“_A-Z(])',
                     paragraph)
    return [p for p in parts if p]


def load_passages(tok, max_tokens: int = PASSAGE_TOKENS, path: Path = STORY_PATH) -> list[str]:
    """Cut the story into passages of at most max_tokens tokens.

    Sentences are packed in order, across paragraph breaks, so short lines of dialogue end up
    together. Each passage starts with the last sentence of the previous one (when it is
    short), so a fact that sits on a boundary is still whole in one of the two passages.
    """
    text = path.read_text(encoding="utf-8")
    sentences = [s for para in re.split(r"\n\s*\n", text) for s in _sentences(para)]
    passages, current, size = [], [], 0
    for sent in sentences:
        n = len(tok.encode(" " + sent))
        if current and size + n > max_tokens:
            passages.append(" ".join(current))
            last = current[-1]
            last_n = len(tok.encode(" " + last))
            current, size = ([last], last_n) if last_n + n <= max_tokens and last_n < 40 else ([], 0)
        if n > max_tokens:  # one very long sentence: cut it by tokens
            ids = tok.encode(" " + sent)
            for i in range(0, len(ids), max_tokens):
                passages.append(tok.decode(ids[i:i + max_tokens]).strip())
            current, size = [], 0
            continue
        current.append(sent)
        size += n
    if current:
        passages.append(" ".join(current))
    return passages


# ---------------------------------------------------------------------------- BM25

class BM25:
    """Okapi BM25 over a fixed list of passages. k1 limits how much repeating a word helps;
    b controls how much long passages are penalised."""

    def __init__(self, passages: list[str], extra: list[list[str]] | None = None,
                 k1: float = 1.5, b: float = 0.75):
        """extra[i]: more texts that passage i should be found by (they are searched, never shown)."""
        self.passages = passages
        self.k1, self.b = k1, b
        self.tf = [Counter(terms(p)) for p in passages]
        for tf, texts in zip(self.tf, extra or []):
            for text in texts:
                tf.update(terms(text))
        self.lengths = [sum(tf.values()) for tf in self.tf]
        self.avg_len = sum(self.lengths) / len(self.lengths)
        df = Counter(t for tf in self.tf for t in tf)
        n = len(passages)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def scores(self, query: dict[str, float]) -> list[float]:
        """query maps each search term to a weight (1.0 = an ordinary word of the question)."""
        out = []
        for tf, length in zip(self.tf, self.lengths):
            s = 0.0
            for t, w in query.items():
                f = tf.get(t, 0)
                if f:
                    s += w * self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length / self.avg_len))
            out.append(s)
        return out


# ---------------------------------------------------------------------------- conversation

def is_followup(question: str) -> bool:
    """Does the question lean on an earlier one ("How did he die?", "Who made it?")?"""
    return bool(FOLLOWUP_WORDS & set(words(question))) or len(terms(question)) <= 1


@dataclass
class Turn:
    question: str
    answer: str = ""
    passage_ids: list[int] = field(default_factory=list)


class Conversation:
    """Remembers the previous questions so follow-ups are searched in context."""

    # Weight of an earlier question's words, relative to 1.0 for the current question's words.
    # Chosen on data/qa/followups.jsonl. Earlier *answers* are deliberately left out of the
    # search: they did not help, and a wrong model answer would lead the next search astray.
    FOLLOWUP_WEIGHT = 0.3   # the question looks like a follow-up
    NEW_TOPIC_WEIGHT = 0.2  # it does not

    def __init__(self, index: BM25, memory: int = 2):
        self.index = index
        self.memory = memory  # how many earlier turns feed the search
        self.turns: list[Turn] = []

    def query_weights(self, question: str) -> dict[str, float]:
        weights = Counter()
        for t in terms(question):
            weights[t] += 1.0
        history_weight = self.FOLLOWUP_WEIGHT if is_followup(question) else self.NEW_TOPIC_WEIGHT
        for age, turn in enumerate(reversed(self.turns[-self.memory:])):
            w = history_weight * 0.5 ** age  # older turns count less
            for t in set(terms(turn.question)):
                weights[t] += w
        return dict(weights)

    def search(self, question: str, k: int = 3) -> list[tuple[int, float]]:
        """Best k passages as (passage id, score), without recording the turn."""
        scores = self.index.scores(self.query_weights(question))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [(i, scores[i]) for i in ranked[:k]]

    def best_sentences(self, question: str, passage_id: int, n: int = 2) -> str:
        """Answer by quoting: the n best-matching neighbouring sentences of the passage."""
        weights = self.query_weights(question)
        sents = _sentences(self.index.passages[passage_id])

        def score(text: str) -> float:
            return sum(w * self.index.idf.get(t, 0.0) for t, w in weights.items() if t in set(terms(text)))

        if len(sents) <= n:
            return " ".join(sents)
        start = max(range(len(sents) - n + 1), key=lambda i: score(" ".join(sents[i:i + n])))
        return " ".join(sents[start:start + n])

    def add(self, question: str, answer: str, passage_ids: list[int]) -> None:
        self.turns.append(Turn(question, answer, passage_ids))

    def reset(self) -> None:
        self.turns.clear()


def gold_passages(passages: list[str], evidence: list[str]) -> set[int]:
    """Ids of the passages that contain at least one of the evidence quotes."""
    norm = [normalize(p) for p in passages]
    return {i for i, p in enumerate(norm) for q in evidence if normalize(q) and normalize(q) in p}


def fact_statement_texts(passages: list[str]) -> list[list[str]]:
    """For each passage, the reworded fact statements (data/qa/fact_statements.jsonl) of the facts
    it contains. A fact is located through the evidence quotes of its questions. Indexing these
    lets "What illness killed Stroud?" find the passage that says "heart disease" and "died"."""
    qa_dir = ROOT / "data" / "qa"
    evidence: dict[str, list[str]] = {}
    for split in ("train", "val", "test", "probe"):
        for line in (qa_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                evidence.setdefault(row["fact"], []).extend(row.get("evidence", []))
    extra = [[] for _ in passages]
    for line in (qa_dir / "fact_statements.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            s = json.loads(line)
            for i in gold_passages(passages, evidence.get(s["fact"], [])):
                extra[i].append(s["statement"])
    return extra


def build_index(tok, with_statements: bool = True) -> BM25:
    passages = load_passages(tok)
    return BM25(passages, fact_statement_texts(passages) if with_statements else None)


def main() -> None:
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    index = build_index(tok)
    print(f"{len(index.passages)} passages, avg {index.avg_len:.0f} search terms each\n")
    conv = Conversation(index)

    def show(question: str) -> None:
        hits = conv.search(question)
        tag = " (follow-up)" if conv.turns and is_followup(question) else ""
        print(f"Q: {question}{tag}")
        for rank, (i, s) in enumerate(hits, 1):
            print(f"  {rank}. [passage {i}, score {s:.1f}] {index.passages[i][:200]}...")
        conv.add(question, "", [i for i, _ in hits])
        print()

    if len(sys.argv) > 1:
        show(" ".join(sys.argv[1:]))
        return
    print("Ask questions (empty line to quit, 'new' to forget the conversation).")
    while (q := input("> ").strip()):
        if q.lower() == "new":
            conv.reset()
            print("(conversation cleared)\n")
            continue
        show(q)


if __name__ == "__main__":
    main()
