from __future__ import annotations

import numpy as np

from atom_gate.channels import bac
from atom_gate.rate_survival import confusability_edges, maximum_independent_set_milp
from atom_gate.representations import bac_representations
from atom_gate.simulation import top_product_atoms


def test_top_product_atoms_returns_the_ordered_reduced_representation() -> None:
    rep = bac_representations(0.12, 0.28, grid_points=7)[-1]
    ordered, atoms, mass = top_product_atoms(rep, blocklength=3, cumulative_target=0.75, max_atoms=40)
    assert ordered.support_size == rep.reduced().support_size
    assert atoms
    assert 0.0 < mass <= 1.0
    for atom_tuple in atoms:
        assert all(0 <= index < ordered.support_size for index in atom_tuple)


def test_confusability_graph_and_mis_are_well_formed() -> None:
    spec = bac(0.12, 0.28)
    rep = bac_representations(0.12, 0.28, grid_points=7)[0]
    edges, mass, atom_count = confusability_edges(
        rep,
        blocklength=3,
        input_size=2,
        output_size=spec.matrix.shape[1],
        cumulative_target=0.8,
        max_atoms=64,
    )
    assert atom_count > 0
    assert mass >= 0.8 - 1e-12
    assert all(0 <= u < v < 8 for u, v in edges)
    solution = maximum_independent_set_milp(8, edges, time_limit_seconds=5.0)
    assert 1 <= int(solution["lower_bound"]) <= 8
