"""
Data Deduplication

Implements MinHash and SimHash algorithms for fuzzy deduplication.
Uses datasketch library for MinHash when available, fallback to pure Python.
"""

import hashlib
import logging
import re
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DeduplicationConfig:
    """Configuration for deduplication."""
    method: str = "minhash"  # minhash, simhash, exact
    threshold: float = 0.8  # Similarity threshold (0-1)
    num_perm: int = 128  # Number of permutations for MinHash
    ngram_size: int = 3  # N-gram size for shingling
    text_fields: List[str] = field(default_factory=lambda: ["prompt", "response"])
    hash_bits: int = 64  # Number of bits for SimHash


@dataclass
class DeduplicationStats:
    """Statistics from deduplication."""
    total_input: int = 0
    total_output: int = 0
    duplicates_found: int = 0
    duplicate_clusters: int = 0

    @property
    def dedup_rate(self) -> float:
        return self.duplicates_found / max(self.total_input, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_input": self.total_input,
            "total_output": self.total_output,
            "duplicates_found": self.duplicates_found,
            "duplicate_clusters": self.duplicate_clusters,
            "dedup_rate": round(self.dedup_rate * 100, 2),
        }


def _get_text(item: Dict, fields: List[str]) -> str:
    """Extract text from item using specified fields."""
    parts = []
    for f in fields:
        val = item.get(f, "")
        if val:
            parts.append(str(val))
    return " ".join(parts)


def _get_ngrams(text: str, n: int = 3) -> Set[str]:
    """Get character n-grams from text."""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    if len(text) < n:
        return {text}
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def _get_word_ngrams(text: str, n: int = 2) -> Set[str]:
    """Get word n-grams from text."""
    words = text.lower().split()
    if len(words) < n:
        return {" ".join(words)}
    return {" ".join(words[i:i+n]) for i in range(len(words) - n + 1)}


# ---- SimHash Implementation ----

def simhash(text: str, hash_bits: int = 64) -> int:
    """Compute SimHash of a text string."""
    tokens = _get_ngrams(text, 3)
    v = [0] * hash_bits

    for token in tokens:
        h = int(hashlib.md5(token.encode('utf-8')).hexdigest(), 16)
        for i in range(hash_bits):
            if h & (1 << i):
                v[i] += 1
            else:
                v[i] -= 1

    fingerprint = 0
    for i in range(hash_bits):
        if v[i] > 0:
            fingerprint |= (1 << i)

    return fingerprint


def simhash_distance(hash1: int, hash2: int, hash_bits: int = 64) -> int:
    """Compute Hamming distance between two SimHash values."""
    x = hash1 ^ hash2
    distance = 0
    while x:
        distance += 1
        x &= x - 1
    return distance


def simhash_similarity(hash1: int, hash2: int, hash_bits: int = 64) -> float:
    """Compute similarity (0-1) between two SimHash values."""
    distance = simhash_distance(hash1, hash2, hash_bits)
    return 1.0 - (distance / hash_bits)


# ---- MinHash Implementation (pure Python fallback) ----

class MinHashPure:
    """Pure Python MinHash implementation."""

    def __init__(self, num_perm: int = 128, seed: int = 42):
        self.num_perm = num_perm
        import random
        rng = random.Random(seed)
        self._a = [rng.randint(1, 2**32 - 1) for _ in range(num_perm)]
        self._b = [rng.randint(0, 2**32 - 1) for _ in range(num_perm)]
        self._p = 4294967311  # A large prime
        self.hashvalues = [2**32 - 1] * num_perm

    def update(self, token: str):
        h = int(hashlib.sha1(token.encode('utf-8')).hexdigest(), 16) & 0xFFFFFFFF
        for i in range(self.num_perm):
            val = (self._a[i] * h + self._b[i]) % self._p
            if val < self.hashvalues[i]:
                self.hashvalues[i] = val

    def jaccard(self, other: 'MinHashPure') -> float:
        if self.num_perm != other.num_perm:
            raise ValueError("Cannot compare MinHash with different num_perm")
        matches = sum(1 for a, b in zip(self.hashvalues, other.hashvalues) if a == b)
        return matches / self.num_perm


def _create_minhash(text: str, num_perm: int = 128, ngram_size: int = 3):
    """Create MinHash signature for text."""
    ngrams = _get_ngrams(text, ngram_size)

    try:
        from datasketch import MinHash
        m = MinHash(num_perm=num_perm)
        for gram in ngrams:
            m.update(gram.encode('utf-8'))
        return m
    except ImportError:
        m = MinHashPure(num_perm=num_perm)
        for gram in ngrams:
            m.update(gram)
        return m


def _minhash_similarity(m1, m2) -> float:
    """Compute Jaccard similarity between two MinHash objects."""
    try:
        from datasketch import MinHash
        if isinstance(m1, MinHash):
            return m1.jaccard(m2)
    except ImportError:
        pass
    return m1.jaccard(m2)


# ---- Main Deduplication Functions ----

def deduplicate_exact(data: List[Dict], text_fields: List[str]) -> Tuple[List[Dict], DeduplicationStats]:
    """Exact deduplication based on text content."""
    stats = DeduplicationStats(total_input=len(data))
    seen = set()
    result = []

    for item in data:
        text = _get_text(item, text_fields)
        h = hashlib.sha256(text.encode('utf-8')).hexdigest()
        if h not in seen:
            seen.add(h)
            result.append(item)
        else:
            stats.duplicates_found += 1

    stats.total_output = len(result)
    stats.duplicate_clusters = stats.duplicates_found  # Each dup is its own cluster in exact mode
    return result, stats


def deduplicate_minhash(
    data: List[Dict],
    config: DeduplicationConfig,
) -> Tuple[List[Dict], DeduplicationStats]:
    """MinHash-based fuzzy deduplication."""
    stats = DeduplicationStats(total_input=len(data))

    # Compute MinHash for each item
    minhashes = []
    for item in data:
        text = _get_text(item, config.text_fields)
        mh = _create_minhash(text, config.num_perm, config.ngram_size)
        minhashes.append(mh)

    # Find duplicates using pairwise comparison (O(n^2) for small datasets)
    # For large datasets, use LSH (datasketch.MinHashLSH)
    is_duplicate = [False] * len(data)
    clusters = 0

    if len(data) <= 10000:
        # Pairwise comparison for small datasets
        for i in range(len(data)):
            if is_duplicate[i]:
                continue
            found_dup = False
            for j in range(i + 1, len(data)):
                if is_duplicate[j]:
                    continue
                sim = _minhash_similarity(minhashes[i], minhashes[j])
                if sim >= config.threshold:
                    is_duplicate[j] = True
                    stats.duplicates_found += 1
                    found_dup = True
            if found_dup:
                clusters += 1
    else:
        # Use LSH for large datasets
        try:
            from datasketch import MinHashLSH
            lsh = MinHashLSH(threshold=config.threshold, num_perm=config.num_perm)

            for i, mh in enumerate(minhashes):
                try:
                    result = lsh.query(mh)
                    if result:
                        is_duplicate[i] = True
                        stats.duplicates_found += 1
                    else:
                        lsh.insert(str(i), mh)
                except ValueError:
                    # Already exists
                    pass
        except ImportError:
            # Fallback to pairwise comparison with sampling
            logger.warning("datasketch not installed, using sampling-based dedup for large dataset")
            import random
            sample_size = min(10000, len(data))
            indices = list(range(len(data)))
            random.shuffle(indices)

            for idx, i in enumerate(indices[:sample_size]):
                if is_duplicate[i]:
                    continue
                for j in indices[idx+1:idx+100]:  # Compare with next 100 items
                    if is_duplicate[j]:
                        continue
                    sim = _minhash_similarity(minhashes[i], minhashes[j])
                    if sim >= config.threshold:
                        is_duplicate[j] = True
                        stats.duplicates_found += 1

    result = [item for item, dup in zip(data, is_duplicate) if not dup]
    stats.total_output = len(result)
    stats.duplicate_clusters = clusters
    return result, stats


def deduplicate_simhash(
    data: List[Dict],
    config: DeduplicationConfig,
) -> Tuple[List[Dict], DeduplicationStats]:
    """SimHash-based fuzzy deduplication."""
    stats = DeduplicationStats(total_input=len(data))

    # Compute SimHash for each item
    hashes = []
    for item in data:
        text = _get_text(item, config.text_fields)
        h = simhash(text, config.hash_bits)
        hashes.append(h)

    # Max allowed Hamming distance based on threshold
    max_distance = int((1.0 - config.threshold) * config.hash_bits)

    is_duplicate = [False] * len(data)
    clusters = 0

    for i in range(len(data)):
        if is_duplicate[i]:
            continue
        found_dup = False
        for j in range(i + 1, len(data)):
            if is_duplicate[j]:
                continue
            dist = simhash_distance(hashes[i], hashes[j], config.hash_bits)
            if dist <= max_distance:
                is_duplicate[j] = True
                stats.duplicates_found += 1
                found_dup = True
        if found_dup:
            clusters += 1

    result = [item for item, dup in zip(data, is_duplicate) if not dup]
    stats.total_output = len(result)
    stats.duplicate_clusters = clusters
    return result, stats


def deduplicate(
    data: List[Dict],
    config: DeduplicationConfig = None,
) -> Tuple[List[Dict], DeduplicationStats]:
    """
    Deduplicate data using configured method.

    Args:
        data: List of data items
        config: Deduplication configuration

    Returns:
        (deduplicated_data, stats)
    """
    if config is None:
        config = DeduplicationConfig()

    if config.method == "exact":
        return deduplicate_exact(data, config.text_fields)
    elif config.method == "minhash":
        return deduplicate_minhash(data, config)
    elif config.method == "simhash":
        return deduplicate_simhash(data, config)
    else:
        raise ValueError(f"Unknown deduplication method: {config.method}")
