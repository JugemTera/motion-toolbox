"""Word vectorizer for the T2M text-motion co-embedding evaluator.

Converts text descriptions into GloVe word embeddings + POS one-hot vectors
for use with the T2M text encoder (TextEncoderBiGRUCo).

This module handles:
    1. Tokenization + POS tagging via spacy
    2. Word-to-GloVe-vector lookup (300-d)
    3. POS one-hot encoding with VIP (Very Important Parts-of-speech) overrides
       for motion-relevant words (locations, body parts, objects, actions, descriptions)

The GloVe data files are expected in the following format (from HumanML3D):
    - {prefix}_data.npy: GloVe vectors, shape (vocab_size, 300)
    - {prefix}_words.pkl: List of vocabulary words
    - {prefix}_idx.pkl: Dict mapping word -> index

References:
    - MoMask: https://github.com/EricGuo5513/momask-codes/blob/main/utils/word_vectorizer.py
    - T2M: https://github.com/EricGuo5513/text-to-motion
"""

from __future__ import annotations

import pickle
from os.path import join as pjoin

import numpy as np

# ============================================================================
# POS tag enumeration (15 categories)
# Standard spacy POS tags + 5 VIP categories for motion-relevant words
# ============================================================================

POS_enumerator: dict[str, int] = {
    "VERB": 0,
    "NOUN": 1,
    "DET": 2,
    "ADP": 3,
    "NUM": 4,
    "AUX": 5,
    "PRON": 6,
    "ADJ": 7,
    "ADV": 8,
    "Loc_VIP": 9,
    "Body_VIP": 10,
    "Obj_VIP": 11,
    "Act_VIP": 12,
    "Desc_VIP": 13,
    "OTHER": 14,
}

# VIP word lists -- motion-domain words that get special POS overrides
Loc_list = (
    "left",
    "right",
    "clockwise",
    "counterclockwise",
    "anticlockwise",
    "forward",
    "back",
    "backward",
    "up",
    "down",
    "straight",
    "curve",
)

Body_list = (
    "arm",
    "chin",
    "foot",
    "feet",
    "face",
    "hand",
    "mouth",
    "leg",
    "waist",
    "eye",
    "knee",
    "shoulder",
    "thigh",
)

Obj_list = (
    "stair",
    "dumbbell",
    "chair",
    "window",
    "floor",
    "car",
    "ball",
    "handrail",
    "baseball",
    "basketball",
)

Act_list = (
    "walk",
    "run",
    "swing",
    "pick",
    "bring",
    "kick",
    "put",
    "squat",
    "throw",
    "hop",
    "dance",
    "jump",
    "turn",
    "stumble",
    "dance",
    "stop",
    "sit",
    "lift",
    "lower",
    "raise",
    "wash",
    "stand",
    "kneel",
    "stroll",
    "rub",
    "bend",
    "balance",
    "flap",
    "jog",
    "shuffle",
    "lean",
    "rotate",
    "spin",
    "spread",
    "climb",
)

Desc_list = (
    "slowly",
    "carefully",
    "fast",
    "careful",
    "slow",
    "quickly",
    "happy",
    "angry",
    "sad",
    "happily",
    "angrily",
    "sadly",
)

VIP_dict: dict[str, tuple[str, ...]] = {
    "Loc_VIP": Loc_list,
    "Body_VIP": Body_list,
    "Obj_VIP": Obj_list,
    "Act_VIP": Act_list,
    "Desc_VIP": Desc_list,
}


class WordVectorizer:
    """GloVe word vectorizer with POS one-hot encoding for T2M evaluation.

    Loads pre-trained GloVe vectors from the HumanML3D vocabulary and converts
    word/POS pairs to (word_vector, pos_one_hot) tuples.

    Args:
        meta_root: Path to directory containing GloVe data files.
        prefix: Filename prefix for the GloVe data files (default "our_vab").

    Example:
        >>> vectorizer = WordVectorizer("./glove", prefix="our_vab")
        >>> word_vec, pos_vec = vectorizer["walk/VERB"]
        >>> word_vec.shape  # (300,)
        >>> pos_vec.shape   # (15,)
    """

    def __init__(self, meta_root: str, prefix: str = "our_vab") -> None:
        vectors = np.load(pjoin(meta_root, f"{prefix}_data.npy"))
        words = pickle.load(open(pjoin(meta_root, f"{prefix}_words.pkl"), "rb"))  # noqa: SIM115
        self.word2idx: dict[str, int] = pickle.load(
            open(pjoin(meta_root, f"{prefix}_idx.pkl"), "rb")  # noqa: SIM115
        )
        self.word2vec: dict[str, np.ndarray] = {w: vectors[self.word2idx[w]] for w in words}

    def _get_pos_ohot(self, pos: str) -> np.ndarray:
        """Convert a POS tag string to a one-hot vector."""
        pos_vec = np.zeros(len(POS_enumerator))
        if pos in POS_enumerator:
            pos_vec[POS_enumerator[pos]] = 1
        else:
            pos_vec[POS_enumerator["OTHER"]] = 1
        return pos_vec

    def __len__(self) -> int:
        return len(self.word2vec)

    def __getitem__(self, item: str) -> tuple[np.ndarray, np.ndarray]:
        """Look up a word/POS pair and return (word_vector, pos_one_hot).

        Args:
            item: String in the format "word/POS" (e.g., "walk/VERB").

        Returns:
            Tuple of (word_vector, pos_one_hot) with shapes (300,) and (15,).
        """
        word, pos = item.split("/")
        if word in self.word2vec:
            word_vec = self.word2vec[word]
            vip_pos = None
            for key, values in VIP_dict.items():
                if word in values:
                    vip_pos = key
                    break
            if vip_pos is not None:
                pos_vec = self._get_pos_ohot(vip_pos)
            else:
                pos_vec = self._get_pos_ohot(pos)
        else:
            word_vec = self.word2vec["unk"]
            pos_vec = self._get_pos_ohot("OTHER")
        return word_vec, pos_vec


def tokenize_text(text: str, max_text_len: int = 20) -> list[str]:
    """Tokenize and POS-tag a text description using spacy.

    Produces word/POS pairs suitable for use with WordVectorizer.
    Requires spacy with the 'en_core_web_sm' model installed.

    Args:
        text: Text description to tokenize (e.g., "A person walks forward").
        max_text_len: Maximum number of tokens to keep (default 20).

    Returns:
        List of "word/POS" strings, truncated to max_text_len.

    Raises:
        ImportError: If spacy or en_core_web_sm is not installed.
    """
    try:
        import spacy
    except ImportError:
        raise ImportError(
            "spacy is required for text tokenization. "
            "Install it with: pip install spacy && python -m spacy download en_core_web_sm"
        )

    # Load spacy model (cached after first call)
    if not hasattr(tokenize_text, "_nlp"):
        try:
            tokenize_text._nlp = spacy.load("en_core_web_sm")  # type: ignore[attr-defined]
        except OSError:
            raise ImportError(
                "spacy model 'en_core_web_sm' not found. Install it with: python -m spacy download en_core_web_sm"
            )

    nlp = tokenize_text._nlp  # type: ignore[attr-defined]
    doc = nlp(text)
    tokens = []
    for token in doc:
        word = token.text.lower().strip()
        if word:
            tokens.append(f"{word}/{token.pos_}")
    return tokens[:max_text_len]


def texts_to_embeddings(
    texts: list[str],
    vectorizer: WordVectorizer,
    max_text_len: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a batch of text descriptions to word embeddings and POS one-hots.

    This is the main entry point for preparing text inputs for the T2M text encoder.

    Args:
        texts: List of text descriptions.
        vectorizer: Initialized WordVectorizer instance.
        max_text_len: Maximum number of tokens per text.

    Returns:
        Tuple of:
            - word_embs: Word embeddings, shape (batch, max_text_len, 300).
            - pos_ohots: POS one-hot vectors, shape (batch, max_text_len, 15).
            - cap_lens: Actual caption lengths, shape (batch,).
    """
    batch_size = len(texts)
    word_emb_dim = 300
    pos_dim = len(POS_enumerator)

    word_embs = np.zeros((batch_size, max_text_len, word_emb_dim))
    pos_ohots = np.zeros((batch_size, max_text_len, pos_dim))
    cap_lens = np.zeros(batch_size, dtype=np.int64)

    for i, text in enumerate(texts):
        tokens = tokenize_text(text, max_text_len)
        cap_len = len(tokens)
        cap_lens[i] = max(cap_len, 1)  # Ensure at least length 1

        for j, token in enumerate(tokens):
            word_vec, pos_vec = vectorizer[token]
            word_embs[i, j] = word_vec
            pos_ohots[i, j] = pos_vec

    return word_embs, pos_ohots, cap_lens
