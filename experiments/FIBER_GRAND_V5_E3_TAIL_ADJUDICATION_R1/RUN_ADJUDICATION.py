#!/usr/bin/env python3
from pathlib import Path
import subprocess, sys
root=Path(__file__).resolve().parent
cmd=[sys.executable, str(root/'src'/'adjudicate.py'), '--repo-root', '/home/afazeli2006/atom_coding', '--timing-repeats', '3']
raise SystemExit(subprocess.call(cmd, cwd=root))
