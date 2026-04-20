"""Amino acid mapping utilities."""

from typing import Dict, Optional

# 3-letter to 1-letter mapping
AA3_TO_AA1: Dict[str, str] = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
    # Common alternates / modified residues
    "MSE": "M",  # selenomethionine -> methionine
    "SEC": "C",  # selenocysteine -> cysteine (approx)
    "PYL": "K",  # pyrrolysine -> lysine (approx)
    "HIP": "H",  # protonated histidine
    "HID": "H",  # histidine (delta protonated)
    "HIE": "H",  # histidine (epsilon protonated)
    # Common ambiguous/unknown labels in structures (mapped to something reasonable or skipped downstream)
    "ASX": "D",  # Asp/Asn ambiguous -> treat as Asp
    "GLX": "E",  # Glu/Gln ambiguous -> treat as Glu
    "UNK": "X",  # unknown residue
}

# 1-letter to index (0-19) - Fixed order
AA1_TO_IDX: Dict[str, int] = {
    "A": 0,
    "C": 1,
    "D": 2,
    "E": 3,
    "F": 4,
    "G": 5,
    "H": 6,
    "I": 7,
    "K": 8,
    "L": 9,
    "M": 10,
    "N": 11,
    "P": 12,
    "Q": 13,
    "R": 14,
    "S": 15,
    "T": 16,
    "V": 17,
    "W": 18,
    "Y": 19,
}

# Reverse mapping: index to 1-letter
IDX_TO_AA1: Dict[int, str] = {i: aa for aa, i in AA1_TO_IDX.items()}

# Standard 3-letter order (for reference)
STANDARD_AA3: list = [
    "ALA",
    "CYS",
    "ASP",
    "GLU",
    "PHE",
    "GLY",
    "HIS",
    "ILE",
    "LYS",
    "LEU",
    "MET",
    "ASN",
    "PRO",
    "GLN",
    "ARG",
    "SER",
    "THR",
    "VAL",
    "TRP",
    "TYR",
]


def aa3_to_idx(resname3: str) -> Optional[int]:
    """Convert 3-letter amino acid code to index (0-19).

    Returns None if not recognized or maps to unknown ("X").
    """
    if not resname3:
        return None
    resname3 = resname3.strip().upper()
    aa1 = AA3_TO_AA1.get(resname3)
    if aa1 is None or aa1 == "X":
        return None
    return AA1_TO_IDX.get(aa1)


def aa1_to_idx(aa1: str) -> Optional[int]:
    """Convert 1-letter amino acid code to index (0-19)."""
    if not aa1:
        return None
    aa1 = aa1.strip().upper()
    return AA1_TO_IDX.get(aa1)


def idx_to_aa1(idx: int) -> str:
    """Convert index (0-19) to 1-letter code."""
    return IDX_TO_AA1.get(idx, "X")


def get_all_aa_indices() -> list:
    """Return list of all standard amino acid indices (0-19)."""
    return list(range(20))
