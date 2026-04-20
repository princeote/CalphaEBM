#!/usr/bin/env python3
"""Check which training proteins are biological multimers via RCSB API.

Queries the RCSB REST API for oligomeric state of each PDB entry.
Proteins that are homodimers/trimers etc. but deposited as single chains
(crystallographic symmetry) pass our monomeric filter but have frustrated
interface residues.

Usage:
    python scripts/check_multimers.py --input train_hq.txt --output multimers.txt
"""

import argparse
import json
import sys
import time

import requests


def get_assembly_info(pdb_id):
    """Query RCSB for biological assembly info."""
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.upper()}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()

        # Get assembly count
        assemblies = data.get("rcsb_entry_info", {})
        assembly_count = assemblies.get("assembly_count", 0)

        return {
            "pdb_id": pdb_id,
            "assembly_count": assembly_count,
        }
    except Exception as e:
        return {"pdb_id": pdb_id, "error": str(e)}


def get_assembly_details(pdb_id):
    """Query RCSB for specific assembly details (oligomeric state)."""
    url = f"https://data.rcsb.org/rest/v1/core/assembly/{pdb_id.upper()}/1"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()

        # Oligomeric details
        details = data.get("rcsb_assembly_info", {})
        oligo_count = details.get("polymer_entity_instance_count", 1)

        # Symmetry info
        symmetry = data.get("rcsb_struct_symmetry", [{}])
        if symmetry:
            sym = symmetry[0]
            oligo_state = sym.get("oligomeric_state", "unknown")
            stoichiometry = sym.get("stoichiometry", ["unknown"])
            symbol = sym.get("symbol", "unknown")
            kind = sym.get("kind", "unknown")
        else:
            oligo_state = "unknown"
            stoichiometry = ["unknown"]
            symbol = "unknown"
            kind = "unknown"

        return {
            "pdb_id": pdb_id,
            "polymer_count": oligo_count,
            "oligomeric_state": oligo_state,
            "stoichiometry": stoichiometry,
            "symmetry": symbol,
            "kind": kind,
        }
    except Exception as e:
        return {"pdb_id": pdb_id, "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="train_hq.txt or similar")
    parser.add_argument("--output", default="multimers.txt", help="Output file")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between API calls (seconds)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Read PDB IDs
    with open(args.input) as f:
        pdb_ids = [line.strip().split()[0] for line in f if line.strip() and not line.startswith("#")]

    print(f"Checking {len(pdb_ids)} proteins for multimer status...")

    multimers = []
    monomers = []
    errors = []

    for i, pdb_id in enumerate(pdb_ids):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(pdb_ids)} checked... ({len(multimers)} multimers found)")

        info = get_assembly_details(pdb_id)

        if info is None or "error" in info:
            errors.append(pdb_id)
            if args.verbose:
                print(f"  ERROR: {pdb_id}: {info}")
            time.sleep(args.delay)
            continue

        is_multimer = info["oligomeric_state"] != "Monomer"

        if is_multimer:
            multimers.append(info)
            if args.verbose:
                print(
                    f"  MULTIMER: {pdb_id} — {info['oligomeric_state']} "
                    f"({info['stoichiometry']}, {info['symmetry']})"
                )
        else:
            monomers.append(pdb_id)

        time.sleep(args.delay)

    # Write results
    with open(args.output, "w") as f:
        f.write(f"# Multimer analysis: {len(pdb_ids)} proteins checked\n")
        f.write(f"# Monomers: {len(monomers)}\n")
        f.write(f"# Multimers: {len(multimers)}\n")
        f.write(f"# Errors: {len(errors)}\n")
        f.write(f"#\n")
        f.write(f"# pdb_id  oligomeric_state  stoichiometry  symmetry  kind\n")
        for info in sorted(multimers, key=lambda x: x["pdb_id"]):
            f.write(
                f"{info['pdb_id']}\t{info['oligomeric_state']}\t"
                f"{info['stoichiometry']}\t{info['symmetry']}\t{info['kind']}\n"
            )

    # Also write a clean list for filtering
    multimer_ids_path = args.output.replace(".txt", "_ids.txt")
    with open(multimer_ids_path, "w") as f:
        for info in sorted(multimers, key=lambda x: x["pdb_id"]):
            f.write(f"{info['pdb_id']}\n")

    # Summary
    print(f"\n{'='*60}")
    print(f"  MULTIMER ANALYSIS COMPLETE")
    print(f"  Total:     {len(pdb_ids)}")
    print(f"  Monomers:  {len(monomers)} ({100*len(monomers)/len(pdb_ids):.1f}%)")
    print(f"  Multimers: {len(multimers)} ({100*len(multimers)/len(pdb_ids):.1f}%)")
    print(f"  Errors:    {len(errors)}")
    print(f"{'='*60}")

    if multimers:
        # Count by type
        from collections import Counter

        states = Counter(info["oligomeric_state"] for info in multimers)
        print(f"\n  Multimer breakdown:")
        for state, count in states.most_common():
            print(f"    {state}: {count}")

    print(f"\n  Saved to: {args.output}")
    print(f"  ID list:  {multimer_ids_path}")

    if errors:
        print(f"\n  Failed PDB IDs: {', '.join(errors[:20])}")
        if len(errors) > 20:
            print(f"    ... and {len(errors) - 20} more")


if __name__ == "__main__":
    main()
