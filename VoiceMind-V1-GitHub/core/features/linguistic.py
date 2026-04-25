
import numpy as np, unicodedata, math
from collections import Counter
from dataclasses import dataclass

@dataclass
class LinguisticFeatures:
    semantic_coherence: float
    lexical_entropy: float
    type_token_ratio: float
    pronoun_inconsistency: float
    syntax_tree_depth: float

    def to_ordered_array(self) -> np.ndarray:
        return np.array([
            self.semantic_coherence, self.lexical_entropy,
            self.type_token_ratio,   self.pronoun_inconsistency,
            self.syntax_tree_depth,
        ])

    @staticmethod
    def feature_names():
        return [
            "semantic_coherence", "lexical_entropy", "type_token_ratio",
            "pronoun_inconsistency", "syntax_tree_depth",
        ]

_EN = {"i","me","my","mine","you","your","he","she","it",
       "we","they","him","her","them","his","its","our"}
_HI = {"मैं","मुझे","मेरा","मेरी","तुम","तुम्हारा","आप","वह","वे","हम"}

def extract_linguistic(text, segments, sentence_embedder=None) -> LinguisticFeatures:
    text  = unicodedata.normalize("NFC", text)
    words = text.lower().split()
    n     = max(len(words), 1)

    ttr     = len(set(words)) / n
    freq    = Counter(words)
    probs   = [c/n for c in freq.values()]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)

    prons   = [w for w in words if w in (_EN | _HI)]
    pinc    = (len(set(prons)) / len(prons)) if len(prons) >= 2 else 0.0

    seg_texts = [s["text"] for s in segments if s.get("text")]
    if sentence_embedder and len(seg_texts) >= 2:
        embs = sentence_embedder.encode(seg_texts)
        sims = []
        for i in range(len(embs) - 1):
            cos = np.dot(embs[i], embs[i+1]) / (
                np.linalg.norm(embs[i]) * np.linalg.norm(embs[i+1]) + 1e-8)
            sims.append(cos)
        coherence = float(np.mean(sims))
    else:
        coherence = 0.0

    syn = float(min(
        np.mean([len(s.split()) for s in seg_texts]) / 5.0, 5.0
    )) if seg_texts else 0.0

    return LinguisticFeatures(
        semantic_coherence=coherence, lexical_entropy=entropy,
        type_token_ratio=ttr, pronoun_inconsistency=pinc,
        syntax_tree_depth=syn,
    )
