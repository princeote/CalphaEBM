# src/calphaebm/data/__init__.py

"""Data loading and processing for PDB structures."""

from calphaebm.data.aa_map import AA1_TO_IDX, AA3_TO_AA1, aa3_to_idx
from calphaebm.data.build_pdb70_like import build_pdb70_like
from calphaebm.data.id_utils import normalize_to_entry_ids, split_entity_id
from calphaebm.data.pdb_chain_dataset import PDBChainDataset
from calphaebm.data.pdb_dataset import PDBSegmentDataset
from calphaebm.data.pdb_parse import (
    download_cif,
    get_residue_sequence,
    iter_fixed_windows,
    load_pdb_segments,
    parse_cif_ca_chains,
    split_chain_on_gaps,
)
from calphaebm.data.rcsb_query import (
    AssemblyInfo,
    PolymerEntityInfo,
    filter_multimers,
    get_assembly_info_batch,
    graphql_polymer_entities_for_entries,
    is_protein_only_entry,
    search_entries_xray_resolution,
)
from calphaebm.data.synthetic import make_extended_chain, random_sequence

__all__ = [
    # Amino acid mapping
    "aa3_to_idx",
    "AA3_TO_AA1",
    "AA1_TO_IDX",
    # ID utilities
    "split_entity_id",
    "normalize_to_entry_ids",
    # PDB parsing
    "download_cif",
    "parse_cif_ca_chains",
    "split_chain_on_gaps",
    "iter_fixed_windows",
    "load_pdb_segments",
    "get_residue_sequence",
    # Dataset
    "PDBSegmentDataset",
    "PDBChainDataset",
    # RCSB API
    "search_entries_xray_resolution",
    "graphql_polymer_entities_for_entries",
    "is_protein_only_entry",
    "PolymerEntityInfo",
    "AssemblyInfo",
    "get_assembly_info_batch",
    "filter_multimers",
    # Dataset building
    "build_pdb70_like",
    # Synthetic data
    "make_extended_chain",
    "random_sequence",
]
