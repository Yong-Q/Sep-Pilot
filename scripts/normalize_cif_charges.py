#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


INCUBATOR_SCRIPTS = (
    Path(__file__).resolve().parents[1] / "new_skill_test" / "molecule_forcefield_pipeline" / "scripts"
)
if str(INCUBATOR_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(INCUBATOR_SCRIPTS))

from normalize_cif_charges import main


if __name__ == "__main__":
    main()
