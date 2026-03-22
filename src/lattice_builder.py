"""
lattice_builder.py
------------------
Lattice-based ASR evaluation system.

A lattice is a compact representation of multiple possible word sequences
(from different ASR models or beam-search hypotheses). Each lattice node
holds a set of alternative words at a given position.

The lattice-based WER allows flexible matching: if any alternative at a
position matches the reference word, it counts as a correct match ("hit"),
avoiding unfair penalties when multiple models agree on a form that differs
from the reference.

Components:
  - LatticeNode: one position with multiple alternatives
  - Lattice: ordered sequence of LatticeNodes
  - LatticeBuilder: aligns multiple hypotheses into a Lattice
  - LatticeWERComputer: computes WER against a Lattice
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LatticeNode:
    """
    Represents one time position in a word lattice.

    Attributes:
        position: Zero-based column index in the lattice.
        alternatives: Set of alternative word forms at this position.
            May include the empty string '' to model optional words
            (insertions in one hypothesis).
    """
    position: int
    alternatives: Set[str] = field(default_factory=set)

    def add(self, word: str) -> None:
        self.alternatives.add(word)

    def matches(self, word: str, case_sensitive: bool = False) -> bool:
        """
        Check if `word` matches any alternative in this node.

        Args:
            word: Word to look for.
            case_sensitive: Whether to use case-sensitive comparison.

        Returns:
            True if the word (or its case-normalised form) appears in alternatives.
        """
        if not case_sensitive:
            word = word.lower()
            return any(a.lower() == word for a in self.alternatives)
        return word in self.alternatives

    def best_alternative(self) -> str:
        """Return the first (arbitrary) alternative as a fallback string."""
        return next(iter(self.alternatives), "")


@dataclass
class Lattice:
    """
    An ordered sequence of LatticeNodes representing a word lattice.

    Attributes:
        nodes: Ordered list of LatticeNode objects.
        metadata: Optional key-value metadata (e.g., source models).
    """
    nodes: List[LatticeNode] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.nodes)

    def __getitem__(self, idx: int) -> LatticeNode:
        return self.nodes[idx]

    def as_sequence(self) -> List[str]:
        """Return a list with the first alternative at each position."""
        return [node.best_alternative() for node in self.nodes]

    def consensus_sequence(self, hypotheses: List[List[str]]) -> List[str]:
        """
        Return the majority-vote sequence across the original hypotheses
        aligned to this lattice.

        Args:
            hypotheses: Original list of tokenised hypotheses.

        Returns:
            List of words chosen by majority vote at each lattice position.
        """
        consensus: List[str] = []
        for node in self.nodes:
            votes: Dict[str, int] = {}
            for hyp in hypotheses:
                for alt in node.alternatives:
                    if alt in hyp:
                        votes[alt] = votes.get(alt, 0) + 1
            if votes:
                winner = max(votes, key=votes.__getitem__)
                consensus.append(winner)
            else:
                consensus.append(node.best_alternative())
        return consensus


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class LatticeBuilder:
    """
    Aligns multiple ASR hypothesis strings into a single Lattice using
    dynamic-programming sequence alignment (similar to multiple sequence
    alignment but simplified for word lattices).

    Strategy:
      1. Choose the longest hypothesis as the "skeleton" (backbone).
      2. Align each additional hypothesis to the backbone using
         pairwise DP alignment.
      3. At each aligned position, add the hypothesis's word as an
         alternative in the corresponding LatticeNode.
      4. Insertions in a hypothesis create a new node with '' in all
         other positions.

    Args:
        insertion_penalty: DP gap-open penalty for insertions.
        deletion_penalty: DP gap-open penalty for deletions.
        substitution_penalty: DP mismatch penalty.
    """

    def __init__(
        self,
        insertion_penalty: float = 1.0,
        deletion_penalty: float = 1.0,
        substitution_penalty: float = 1.0,
    ):
        self.insertion_penalty = insertion_penalty
        self.deletion_penalty = deletion_penalty
        self.substitution_penalty = substitution_penalty

    # ------------------------------------------------------------------
    # Pairwise alignment (Needleman-Wunsch style)
    # ------------------------------------------------------------------

    def _align(
        self,
        seq_a: List[str],
        seq_b: List[str],
    ) -> Tuple[List[str], List[str]]:
        """
        Globally align two word sequences using a simplified NW algorithm.

        Returns:
            Two aligned sequences of equal length where gaps are represented
            by the empty string ''.
        """
        m, n = len(seq_a), len(seq_b)

        # Fill DP table
        dp = np.zeros((m + 1, n + 1))
        for i in range(1, m + 1):
            dp[i][0] = dp[i - 1][0] + self.deletion_penalty
        for j in range(1, n + 1):
            dp[0][j] = dp[0][j - 1] + self.insertion_penalty

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                match_cost = 0.0 if seq_a[i - 1] == seq_b[j - 1] else self.substitution_penalty
                dp[i][j] = min(
                    dp[i - 1][j - 1] + match_cost,   # substitution / match
                    dp[i - 1][j] + self.deletion_penalty,   # delete from a
                    dp[i][j - 1] + self.insertion_penalty,  # insert from b
                )

        # Traceback
        aligned_a: List[str] = []
        aligned_b: List[str] = []
        i, j = m, n
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                match_cost = 0.0 if seq_a[i - 1] == seq_b[j - 1] else self.substitution_penalty
                if dp[i][j] == dp[i - 1][j - 1] + match_cost:
                    aligned_a.insert(0, seq_a[i - 1])
                    aligned_b.insert(0, seq_b[j - 1])
                    i -= 1
                    j -= 1
                elif dp[i][j] == dp[i - 1][j] + self.deletion_penalty:
                    aligned_a.insert(0, seq_a[i - 1])
                    aligned_b.insert(0, "")
                    i -= 1
                else:
                    aligned_a.insert(0, "")
                    aligned_b.insert(0, seq_b[j - 1])
                    j -= 1
            elif i > 0:
                aligned_a.insert(0, seq_a[i - 1])
                aligned_b.insert(0, "")
                i -= 1
            else:
                aligned_a.insert(0, "")
                aligned_b.insert(0, seq_b[j - 1])
                j -= 1

        return aligned_a, aligned_b

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, hypotheses: List[str]) -> Lattice:
        """
        Build a Lattice from multiple hypothesis strings.

        Args:
            hypotheses: List of hypothesis strings (one per model/beam).

        Returns:
            Lattice object representing the alignment.
        """
        if not hypotheses:
            raise ValueError("Need at least one hypothesis to build a lattice.")

        # Tokenise all hypotheses
        tokenised: List[List[str]] = [h.split() for h in hypotheses]

        # Use the longest as backbone
        backbone_idx = max(range(len(tokenised)), key=lambda i: len(tokenised[i]))
        backbone = tokenised[backbone_idx]

        # Initialise lattice with backbone nodes
        lattice = Lattice(
            nodes=[
                LatticeNode(position=i, alternatives={w})
                for i, w in enumerate(backbone)
            ],
            metadata={"backbone_index": backbone_idx, "num_hypotheses": len(hypotheses)},
        )

        # Align each other hypothesis to the backbone
        for idx, hyp_tokens in enumerate(tokenised):
            if idx == backbone_idx:
                continue

            aligned_backbone, aligned_hyp = self._align(backbone, hyp_tokens)

            # Merge aligned_hyp alternatives into lattice nodes
            # We need a mapping from aligned positions → lattice node positions
            lattice_pos = 0
            expanded_nodes: List[Optional[LatticeNode]] = list(lattice.nodes)

            new_nodes: List[LatticeNode] = []
            node_ptr = 0
            for aligned_b, aligned_h in zip(aligned_backbone, aligned_hyp):
                if aligned_b != "":
                    # Existing lattice position
                    if node_ptr < len(lattice.nodes):
                        node = lattice.nodes[node_ptr]
                        if aligned_h != "":
                            node.add(aligned_h)
                        new_nodes.append(node)
                        node_ptr += 1
                else:
                    # Insertion in hyp → new lattice node
                    new_node = LatticeNode(
                        position=len(new_nodes),
                        alternatives={"", aligned_h} if aligned_h else {""},
                    )
                    new_nodes.append(new_node)

            # Re-number positions
            for p, node in enumerate(new_nodes):
                node.position = p

            lattice.nodes = new_nodes

        logger.info(
            "Lattice built: %d positions from %d hypotheses.",
            len(lattice), len(hypotheses),
        )
        return lattice


# ---------------------------------------------------------------------------
# Lattice WER Computer
# ---------------------------------------------------------------------------

class LatticeWERComputer:
    """
    Computes a lattice-aware WER against a reference string.

    At each lattice position, the reference word is matched against ALL
    alternatives.  If any alternative matches, it is counted as a 'hit'
    regardless of which specific form is used (reducing unfair penalties
    when ASR models consistently use a different-but-acceptable form).

    Additionally implements a model-agreement heuristic:
      If all N hypotheses agree on a word that differs from the reference,
      the reference is considered potentially noisy and the penalty is
      **halved** (configurable via `agreement_discount`).

    Args:
        agreement_discount: Fraction to reduce penalty when all models agree.
            0.0 = no discount, 1.0 = no penalty at all.
        min_hypotheses_for_discount: Minimum number of hypotheses that must
            agree before the discount applies.
    """

    def __init__(
        self,
        agreement_discount: float = 0.5,
        min_hypotheses_for_discount: int = 2,
    ):
        self.agreement_discount = agreement_discount
        self.min_hypotheses_for_discount = min_hypotheses_for_discount

    def compute(
        self,
        reference: str,
        lattice: Lattice,
        hypotheses: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compute lattice-based WER for a single sentence.

        Args:
            reference: Ground-truth transcription string.
            lattice: Lattice object.
            hypotheses: Original hypothesis strings (for agreement check).

        Returns:
            Dict with keys:
              - lattice_wer (float)
              - standard_wer (float, computed against first hypothesis)
              - hits, insertions, deletions, substitutions (ints)
              - reference_tokens, lattice_sequence
        """
        ref_tokens = reference.split()
        n_ref = len(ref_tokens)
        n_lat = len(lattice)

        # --- Alignment: ref vs lattice ---
        # Use edit distance where a lattice node "matches" if ANY alternative
        # matches the reference token (flexible matching)

        # DP array
        dp = np.zeros((n_ref + 1, n_lat + 1))
        for i in range(1, n_ref + 1):
            dp[i][0] = i  # deletions
        for j in range(1, n_lat + 1):
            dp[0][j] = j  # insertions

        for i in range(1, n_ref + 1):
            for j in range(1, n_lat + 1):
                node = lattice[j - 1]
                ref_word = ref_tokens[i - 1]
                is_match = node.matches(ref_word)

                # Apply agreement discount
                sub_cost = 0.0 if is_match else 1.0
                if not is_match and hypotheses is not None:
                    hyp_tokens_at_j = []
                    for hyp in hypotheses:
                        toks = hyp.split()
                        if j - 1 < len(toks):
                            hyp_tokens_at_j.append(toks[j - 1])
                    if (
                        len(hyp_tokens_at_j) >= self.min_hypotheses_for_discount
                        and len(set(hyp_tokens_at_j)) == 1
                    ):
                        # All models agree on a form different from reference
                        sub_cost = 1.0 - self.agreement_discount

                dp[i][j] = min(
                    dp[i - 1][j - 1] + sub_cost,   # match / substitution
                    dp[i - 1][j] + 1.0,             # deletion
                    dp[i][j - 1] + 1.0,             # insertion
                )

        # Total weighted edit distance
        edit_distance = dp[n_ref][n_lat]
        lattice_wer = float(edit_distance) / max(n_ref, 1)

        # Standard WER (first hypothesis vs reference)
        first_hyp = lattice.as_sequence()
        first_hyp_str = " ".join(w for w in first_hyp if w)
        try:
            from jiwer import wer as _wer
            standard_wer = float(_wer([reference], [first_hyp_str]))
        except Exception:
            standard_wer = float("nan")

        return {
            "lattice_wer": round(lattice_wer, 6),
            "standard_wer": round(standard_wer, 6),
            "edit_distance": round(float(edit_distance), 4),
            "reference_length": n_ref,
            "lattice_length": n_lat,
            "reference_tokens": ref_tokens,
            "lattice_sequence": lattice.as_sequence(),
        }

    def compute_corpus(
        self,
        references: List[str],
        lattices: List[Lattice],
        hypotheses_list: Optional[List[List[str]]] = None,
    ) -> Dict[str, Any]:
        """
        Corpus-level lattice WER computation.

        Args:
            references: List of reference strings.
            lattices: List of Lattice objects (one per utterance).
            hypotheses_list: List of hypothesis lists (one per utterance).

        Returns:
            Dict with corpus-level lattice_wer, standard_wer, and per-
            utterance results.
        """
        if len(references) != len(lattices):
            raise ValueError(
                f"Length mismatch: {len(references)} references vs "
                f"{len(lattices)} lattices."
            )

        total_edit_distance = 0.0
        total_ref_length = 0
        per_utterance: List[Dict] = []

        for i, (ref, lat) in enumerate(zip(references, lattices)):
            hyps = hypotheses_list[i] if hypotheses_list else None
            result = self.compute(ref, lat, hyps)
            total_edit_distance += result["edit_distance"]
            total_ref_length += result["reference_length"]
            per_utterance.append(result)

        corpus_wer = total_edit_distance / max(total_ref_length, 1)
        corpus_standard_wer = np.mean([r["standard_wer"] for r in per_utterance
                                        if not np.isnan(r["standard_wer"])])

        logger.info(
            "Corpus lattice WER: %.4f | Standard WER: %.4f | Utterances: %d",
            corpus_wer, corpus_standard_wer, len(references),
        )

        return {
            "corpus_lattice_wer": round(corpus_wer, 6),
            "corpus_standard_wer": round(float(corpus_standard_wer), 6),
            "num_utterances": len(references),
            "per_utterance": per_utterance,
        }


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_and_evaluate(
    hypotheses: List[str],
    reference: str,
    agreement_discount: float = 0.5,
) -> Dict[str, Any]:
    """
    One-shot helper: build lattice from hypotheses and compute WER.

    Args:
        hypotheses: List of ASR hypothesis strings.
        reference: Ground-truth reference string.
        agreement_discount: Discount when all models agree.

    Returns:
        WER result dict plus the lattice bins for inspection.
    """
    builder = LatticeBuilder()
    computer = LatticeWERComputer(agreement_discount=agreement_discount)

    lattice = builder.build(hypotheses)
    result = computer.compute(reference, lattice, hypotheses)

    # Add human-readable lattice bins
    result["lattice_bins"] = [sorted(node.alternatives) for node in lattice.nodes]

    return result
