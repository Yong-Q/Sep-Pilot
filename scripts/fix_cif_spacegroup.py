#!/usr/bin/env python3
"""Fix CIF space group definitions for RASPA compatibility.

RASPA expects _symmetry_space_group_name_H-M (not _alt variant).
This script normalizes PACMAN-generated CIFs.
"""
import sys
import re
from pathlib import Path


def fix_cif_spacegroup(cif_path: str) -> bool:
    path = Path(cif_path)
    content = path.read_text(encoding="utf-8")
    
    # Check if already has correct format
    if "_symmetry_space_group_name_H-M" in content and "_space_group_name_H-M_alt" not in content:
        print(f"  ✅ {path.name}: already correct format")
        return False
    
    # Replace _space_group_name_H-M_alt with _symmetry_space_group_name_H-M
    if "_space_group_name_H-M_alt" in content:
        content = content.replace(
            "_space_group_name_H-M_alt",
            "_symmetry_space_group_name_H-M"
        )
        print(f"  🔧 {path.name}: fixed _space_group_name_H-M_alt → _symmetry_space_group_name_H-M")
    
    # Replace _space_group_IT_number with _symmetry_Int_Tables_number
    if "_space_group_IT_number" in content:
        content = content.replace(
            "_space_group_IT_number",
            "_symmetry_Int_Tables_number"
        )
        print(f"  🔧 {path.name}: fixed _space_group_IT_number → _symmetry_Int_Tables_number")
    
    # Add _symmetry_generation if missing
    if "_symmetry_generation" not in content and "_space_group_symop_operation_xyz" in content:
        # Replace loop_ header
        content = content.replace(
            "_space_group_symop_operation_xyz",
            "_symmetry_equiv_pos_as_xyz"
        )
        print(f"  🔧 {path.name}: fixed _space_group_symop → _symmetry_equiv_pos")
    
    path.write_text(content, encoding="utf-8")
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: fix_cif_spacegroup.py <cif_path_or_dir>")
        sys.exit(1)
    
    target = Path(sys.argv[1])
    if target.is_dir():
        for cif in sorted(target.glob("**/*.cif")):
            fix_cif_spacegroup(str(cif))
    else:
        fix_cif_spacegroup(str(target))


if __name__ == "__main__":
    main()
