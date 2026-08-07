import json, math
from pathlib import Path

def test_shell3_count():
    assert 2*64*sum(math.comb(63,i) for i in range(4)) == 5_341_184

def test_targets():
    root=Path(__file__).resolve().parents[1]
    d=json.loads((root/'config'/'targets.json').read_text())
    assert len(d['targets']) == 7
    assert all(t[1] in (0.005,0.01) for t in d['targets'])
