"""test/python/fixtures.py — the published MACE model, its readout basis and a few structures, built once."""
import functools

import numpy as np
from ase.build import bulk

import harness  # noqa: F401  (sets OUT, THREADS, sys.path before lib is imported)
import lib


@functools.lru_cache(maxsize=None)
def calc():
    return lib.load_calc()


@functools.lru_cache(maxsize=None)
def basis():
    return lib.Basis(lib.load_calc())     # its own calculator: Basis hooks the model it is given


def rattled(n=2, sigma=0.03, seed=0):
    """An n×n×n cubic fcc Al supercell at A0 with seeded random displacements (off equilibrium: forces ≠ 0)."""
    at = bulk("Al", "fcc", a=lib.A0, cubic=True).repeat((n, n, n))
    at.positions += sigma * np.random.default_rng(seed).standard_normal(at.positions.shape)
    return at


def mace_E_F(atoms, c=None):
    x = atoms.copy()
    x.calc = c or calc()
    return x.get_potential_energy(), x.get_forces()
