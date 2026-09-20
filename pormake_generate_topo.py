#!/usr/bin/env python3
"""
PORMAKE unified generator for MOF/COF/HOF structures with topology filtering.

Usage:
    source activate pormake
    python pormake_generate_topo.py generate --type HOF --n 42 --output-dir ./u_hof_structures --topo utk hxg bto cds cdt srs eta
    python pormake_generate_topo.py generate --type MOF --n 100 --output-dir ./mof_output
    python pormake_generate_topo.py list --type HOF --show-samples
    python pormake_generate_topo.py rmsd --type MOF
"""
import sys
import pickle
import random
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from itertools import chain

import pormake as pm
from pormake import Database, BuildingBlock, Topology, Locator

pm.log.disable_print()
pm.log.disable_file_print()

# ── Constants ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/user/gcmc_agent")
DB_DIR = PROJECT_ROOT / "pormake_db"

BB_DIRS = {
    "MOF": DB_DIR / "bbs_MOF",
    "COF": DB_DIR / "bbs_COF",
    "HOF": DB_DIR / "bbs",
}

FAILED_TOPOS = [
    'ibb', 'ken', 'mmo', 'css', 'tfy-a', 'elv', 'tsn', 'lcz', 'xbn', 'dgo',
    'ten', 'scu-h', 'zim', 'ild', 'cds-t', 'crt', 'jsm', 'rht-x', 'mab', 'ddi',
    'mhq', 'nbo-x', 'tcb', 'zst', 'she-d', 'ffg-a', 'cdh', 'ast-d', 'ffj-a',
    'ddy', 'llw-z', 'tpt', 'utx', 'fnx', 'roa', 'nia-d', 'dnf-a',
    'lcw_component_3', 'baz-a', 'yzh', 'dia-x',
]


# ── RMSD pickle management ────────────────────────────────────────────────────
def get_rmsd_pickle_path(material_type: str) -> Path:
    return DB_DIR / f"rmsd_{material_type.lower()}.pickle"


def _compute_one_topo(args):
    """Worker function: compute RMSD-filtered nodes for a single topology."""
    topo_name, bb_dir_str, topo_dir_str, node_bb_names = args
    bb_dir = Path(bb_dir_str)
    topo_dir = Path(topo_dir_str) if topo_dir_str else None

    import pormake as pm
    from pormake import Database, Locator
    pm.log.disable_print()
    pm.log.disable_file_print()

    db = Database(bb_dir=bb_dir, topo_dir=topo_dir)
    try:
        topo = db.get_topo(topo_name)
    except Exception:
        return topo_name, []

    total = []
    for j, cn in enumerate(topo.unique_cn):
        node_candidates = [n for n in node_bb_names
                          if db.get_bb(n).n_connection_points == cn]
        loc = Locator()
        for node_bb in list(node_candidates):
            bb_xyz = db.get_bb(node_bb)
            rmsd = loc.calculate_rmsd(topo.unique_local_structures[j], bb_xyz)
            if rmsd > 0.3:
                node_candidates.remove(node_bb)
        total.append(node_candidates)

    return topo_name, total


def generate_rmsd_pickle(material_type: str, n_cores: int = 1, topo_dir: Optional[Path] = None) -> Dict:
    """Generate rmsd_calculated_node.pickle for a given material type."""
    bb_dir = BB_DIRS[material_type]
    db = Database(bb_dir=bb_dir, topo_dir=topo_dir)

    bbs = db._get_bb_list()
    node_bbs = [f for f in bbs if f.startswith('N') or f.startswith('C')]

    topos = db._get_topology_list()
    topos = list(set(topos).difference(set(FAILED_TOPOS)))

    topo_dir_str = str(topo_dir) if topo_dir else None
    tasks = [(t, str(bb_dir), topo_dir_str, node_bbs) for t in topos]

    topo_dict = {}
    print(f"[{material_type}] Computing RMSD for {len(topos)} topos × {len(node_bbs)} nodes, cores={n_cores}...")

    if n_cores <= 1:
        for i, task in enumerate(tasks):
            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{len(topos)}")
            name, result = _compute_one_topo(task)
            topo_dict[name] = result
    else:
        from multiprocessing import Pool
        with Pool(n_cores) as pool:
            for i, (name, result) in enumerate(pool.imap_unordered(_compute_one_topo, tasks)):
                topo_dict[name] = result
                if (i + 1) % 20 == 0:
                    print(f"  Progress: {i+1}/{len(topos)}")

    pickle_path = get_rmsd_pickle_path(material_type)
    pickle_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pickle_path, 'wb') as f:
        pickle.dump(topo_dict, f)
    print(f"[{material_type}] Saved: {pickle_path}")

    return topo_dict


def load_rmsd_pickle(material_type: str) -> Dict:
    pickle_path = get_rmsd_pickle_path(material_type)
    if pickle_path.exists():
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)
    print(f"[{material_type}] No cached rmsd pickle, generating...")
    return generate_rmsd_pickle(material_type)


# ── Helpers ────────────────────────────────────────────────────────────────────
def count_atoms_in_bb(bb: BuildingBlock) -> int:
    if bb is None:
        return 0
    return np.sum(bb.atoms.get_chemical_symbols() != 'X')


def calculate_n_atoms(topo: Topology, nodes: dict, edges: dict) -> int:
    nt_counts = {nt: np.sum(topo.node_types == nt) for nt in topo.unique_node_types}
    et_counts = {tuple(et): np.sum(np.all(topo.edge_types == et[np.newaxis, :], axis=1))
                 for et in topo.unique_edge_types}
    counts = 0
    for nt, node in nodes.items():
        counts += nt_counts[nt] * count_atoms_in_bb(node)
    for et, edge in edges.items():
        counts += et_counts[et] * count_atoms_in_bb(edge)
    return counts


def make_name(topo: Topology, nodes: dict, edges: dict) -> str:
    en = lambda x: x.name if x else "E0"
    node_names = [bb.name for bb in nodes.values()]
    edge_names = [en(bb) for bb in edges.values()]
    return "+".join([topo.name] + node_names + edge_names)


def has_metal(nodes: dict, edges: dict) -> bool:
    for bb in chain(nodes.values(), edges.values()):
        if bb is not None and bb.has_metal:
            return True
    return False


# ── Candidate generation ──────────────────────────────────────────────────────
def generate_candidates(
    material_type: str,
    n: int,
    max_atoms: int = 1500,
    topo_dir: Optional[Path] = None,
    bb_dir: Optional[Path] = None,
    topo_list: Optional[List[str]] = None,  # New parameter for topology filtering
) -> List[str]:
    if bb_dir is None:
        bb_dir = BB_DIRS[material_type]
    db = Database(bb_dir=bb_dir, topo_dir=topo_dir)
    rmsd_data = load_rmsd_pickle(material_type)

    valid_topos = [t for t in rmsd_data if all(len(x) > 0 for x in rmsd_data[t])]
    topos = list(set(valid_topos).difference(set(FAILED_TOPOS)))
    
    # Filter by specified topologies if provided
    if topo_list:
        topos = [t for t in topos if t in topo_list]
        print(f"Filtering to specified topologies: {topo_list}")
        print(f"Found {len(topos)} valid topologies from specified list")

    edge_bbs = [f for f in db._get_bb_list() if f.startswith('E') or f.startswith('L')] + ['E0']
    want_metal = (material_type == "MOF")

    candidates = []
    attempts = 0
    max_attempts = n * 100

    while len(candidates) < n and attempts < max_attempts:
        attempts += 1
        topo_name = random.choice(topos)
        try:
            assert topo_name in rmsd_data
            topo = db.get_topo(topo_name)
        except:
            continue

        is_valid = True
        for node_info in rmsd_data[topo_name]:
            if node_info == []:
                is_valid = False
                topos.remove(topo_name)
                break
        if not is_valid:
            continue

        nodes = {k: db.get_bb(random.choice(v))
                 for k, v in zip(topo.unique_node_types, rmsd_data[topo_name])}
        edges = {}
        for k in topo.unique_edge_types:
            re = random.choice(edge_bbs)
            edges[tuple(k)] = db.get_bb(re) if re != 'E0' else None

        if want_metal and not has_metal(nodes, edges):
            continue

        n_atoms = calculate_n_atoms(topo, nodes, edges)
        if n_atoms > max_atoms:
            continue

        name = make_name(topo, nodes, edges)
        if name not in candidates:
            candidates.append(name)

    return candidates


# ── Structure building ─────────────────────────────────────────────────────────
def name_to_structure(name: str, db: Database):
    tokens = name.split("+")
    topo_name = tokens[0]

    node_names = []
    edge_names = []
    for bb in tokens[1:]:
        if bb.startswith("N") or bb.startswith("C"):
            node_names.append(bb)
        if bb.startswith("E") or bb.startswith("L"):
            edge_names.append(bb)

    topo = db.get_topo(topo_name)
    node_bbs = [db.get_bb(f'{n}.xyz') for n in node_names]
    edge_bbs = {
        tuple(et): None if n == 'E0' else db.get_bb(f'{n}.xyz')
        for et, n in zip(topo.unique_edge_types, edge_names)
    }

    builder = pm.Builder()
    return builder.build_by_type(topo, node_bbs, edge_bbs)


def build_and_save(
    candidates: List[str],
    material_type: str,
    output_dir: Path,
    cutoff: float = 45.0,
    topo_dir: Optional[Path] = None,
):
    bb_dir = BB_DIRS[material_type]
    db = Database(bb_dir=bb_dir, topo_dir=topo_dir)

    small_dir = output_dir / "small"
    large_dir = output_dir / "large"
    small_dir.mkdir(parents=True, exist_ok=True)
    large_dir.mkdir(parents=True, exist_ok=True)

    success = failed = skipped = 0

    for i, name in enumerate(candidates):
        print(f"[{i+1}/{len(candidates)}] {name[:60]}...", end=" ")
        try:
            structure = name_to_structure(name, db)
            if isinstance(structure, str):
                print("skip (invalid)")
                skipped += 1
                continue

            min_cell = np.min(structure.atoms.cell.cellpar()[:3])
            if min_cell < 4.5:
                print("skip (small cell)")
                skipped += 1
                continue

            max_cell = np.max(structure.atoms.cell.cellpar()[:3])
            if max_cell < cutoff:
                cif_path = small_dir / f"{name}.cif"
                structure.write_cif(str(cif_path))
            else:
                cif_path = large_dir / f"{name}.cif"
                structure.write_cif(str(cif_path))

            if cif_path.exists():
                print("OK (small)" if max_cell < cutoff else "OK (large)")
                success += 1
            else:
                print("FAIL (CIF write error)")
                failed += 1
        except Exception as e:
            print(f"FAIL: {e}")
            failed += 1

    print(f"\nDone: {success} success, {failed} failed, {skipped} skipped")
    print(f"Output: {output_dir}")
    return success, failed, skipped


# ── CLI commands ───────────────────────────────────────────────────────────────
def cmd_generate(args):
    output_dir = Path(args.output_dir)
    target = args.n
    if args.small:
        args.max_atoms = min(args.max_atoms, 1000)
        args.cutoff = min(args.cutoff, 30.0)
        if args.max_cell is None:
            args.max_cell = 30.0
    max_cell_limit = args.max_cell

    # Use custom bb_dir if provided, otherwise use default
    bb_dir = Path(args.bb_dir) if args.bb_dir else BB_DIRS[args.type]

    print(f"[{args.type}] Generating {target} structures, max_atoms={args.max_atoms}, cutoff={args.cutoff}, max_cell={max_cell_limit}")
    print(f"  bb_dir: {bb_dir}")
    print(f"  output: {output_dir}")
    if args.topo:
        print(f"  specified topologies: {args.topo}")

    output_dir.mkdir(parents=True, exist_ok=True)
    small_dir = output_dir / "small"
    large_dir = output_dir / "large"
    small_dir.mkdir(parents=True, exist_ok=True)
    large_dir.mkdir(parents=True, exist_ok=True)
    cutoff = args.cutoff
    db = Database(bb_dir=bb_dir)
    rmsd_data = load_rmsd_pickle(args.type)

    total_success = 0
    total_failed = 0
    tried = set()
    round_num = 0

    while total_success < target:
        round_num += 1
        need = target - total_success
        batch_size = max(need * 3, 10)
        print(f"\n--- Round {round_num}: need {need} more, generating {batch_size} candidates ---")

        candidates = generate_candidates(
            material_type=args.type,
            n=batch_size,
            max_atoms=args.max_atoms,
            bb_dir=bb_dir,
            topo_list=args.topo,  # Pass topology list
        )
        # skip already-tried candidates
        candidates = [c for c in candidates if c not in tried]
        print(f"[{args.type}] Got {len(candidates)} new candidates")

        if not candidates:
            print("No new candidates available, stopping.")
            break

        for i, name in enumerate(candidates):
            if total_success >= target:
                break
            tried.add(name)
            print(f"[{total_success+1}/{target}] {name[:60]}...", end=" ")
            try:
                structure = name_to_structure(name, db)
                if isinstance(structure, str):
                    print("skip (invalid)")
                    continue

                min_cell = np.min(structure.atoms.cell.cellpar()[:3])
                if min_cell < 4.5:
                    print("skip (small cell)")
                    continue

                max_cell = np.max(structure.atoms.cell.cellpar()[:3])
                if max_cell_limit and max_cell > max_cell_limit:
                    print(f"skip (max_cell={max_cell:.1f} > {max_cell_limit})")
                    continue
                size_class = "small" if max_cell < cutoff else "large"
                if args.selection != "all" and size_class != args.selection:
                    print(f"skip ({size_class}, selection={args.selection})")
                    continue
                if size_class == "small":
                    cif_path = small_dir / f"{name}.cif"
                    structure.write_cif(str(cif_path))
                else:
                    cif_path = large_dir / f"{name}.cif"
                    structure.write_cif(str(cif_path))

                if cif_path.exists():
                    print("OK (small)" if max_cell < cutoff else "OK (large)")
                    total_success += 1
                else:
                    print("FAIL (CIF write error)")
                    total_failed += 1
            except Exception as e:
                print(f"FAIL: {e}")
                total_failed += 1

    # write all tried candidates
    with open(output_dir / "candidates.txt", 'w') as f:
        for c in tried:
            f.write(c + '\n')

    print(f"\nDone: {total_success} success, {total_failed} failed")
    print(f"Output: {output_dir}")


def cmd_list(args):
    bb_dir = BB_DIRS[args.type]
    db = Database(bb_dir=bb_dir)
    bbs = db._get_bb_list()
    topos = db._get_topology_list()

    nodes = [f for f in bbs if f.startswith('N') or f.startswith('C')]
    edges = [f for f in bbs if f.startswith('E') or f.startswith('L')]

    print(f"[{args.type}] Building blocks:")
    print(f"  bb_dir: {bb_dir}")
    print(f"  Nodes: {len(nodes)} (N*/C*)")
    print(f"  Edges: {len(edges)} (E*/L*)")
    print(f"  Topologies: {len(topos)}")

    if args.show_samples:
        print(f"\nSample nodes: {nodes[:10]}")
        print(f"Sample edges: {edges[:10]}")
        print(f"Sample topos: {topos[:10]}")


def cmd_rmsd(args):
    topo_dir = Path(args.topo_dir) if args.topo_dir else None
    generate_rmsd_pickle(args.type, n_cores=args.cores, topo_dir=topo_dir)


def main():
    parser = argparse.ArgumentParser(description="PORMAKE unified generator for MOF/COF/HOF")
    sub = parser.add_subparsers(dest="command")

    gen = sub.add_parser("generate", help="Generate structures")
    gen.add_argument("--type", required=True, choices=["MOF", "COF", "HOF"])
    gen.add_argument("--n", type=int, default=100)
    gen.add_argument("--max-atoms", type=int, default=1500)
    gen.add_argument("--output-dir", default="./pormake_output")
    gen.add_argument("--cutoff", type=float, default=45.0)
    gen.add_argument("--max-cell", type=float, default=None, help="Reject structures with max cell edge > this value (Å)")
    gen.add_argument("--small", action="store_true", help="Small cell mode: atoms<1000, cell<30Å")
    gen.add_argument("--selection", choices=["all", "large", "small"], default="all",
                     help="Count only successfully written structures in this size class toward --n")
    gen.add_argument("--bb-dir", type=str, default=None, help="Custom building block directory (overrides default)")
    gen.add_argument("--topo", nargs="+", type=str, default=None, help="Specify list of topologies to generate (e.g., --topo utk hxg bto cds cdt srs eta)")

    ls = sub.add_parser("list", help="List available BBs and topologies")
    ls.add_argument("--type", required=True, choices=["MOF", "COF", "HOF"])
    ls.add_argument("--show-samples", action="store_true")

    rd = sub.add_parser("rmsd", help="Pre-generate rmsd pickle")
    rd.add_argument("--type", required=True, choices=["MOF", "COF", "HOF"])
    rd.add_argument("--cores", type=int, default=4, help="Number of CPU cores")
    rd.add_argument("--topo-dir", type=str, default=None, help="Custom topology directory (default: pormake built-in)")

    args = parser.parse_args()
    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "rmsd":
        cmd_rmsd(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
