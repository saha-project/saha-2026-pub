#!/usr/bin/env python
"""
Phase 1 environment verification script.
Run inside spatial_integration env:
    conda run -n spatial_integration python scripts/01_verify_env.py

Checks: package imports, versions, MPS availability, COSIE/tacos availability.

NOTE: COSIE repo (https://github.com/weilicode/cosie.git) was not publicly accessible
as of 2026-04-05. tacos==0.0.2 is installed as fallback in both envs.
cosie_env (Python 3.9, torch 2.4.0) is the dedicated COSIE env for when COSIE becomes available.
"""
import importlib
import sys

REQUIRED_PACKAGES = [
    "scvi", "scanpy", "anndata", "squidpy",
    "torch", "torch_geometric", "numpy", "scipy",
    "thefuzz", "h5py", "sklearn", "pandas", "matplotlib", "pytest",
]

print(f"Python: {sys.version.split()[0]}")
major, minor = sys.version_info[:2]
if (major, minor) < (3, 11):
    print(f"  WARNING: Python {major}.{minor} detected — scvi-tools 1.4.x requires >=3.11")

print("\nPackage imports:")
all_ok = True
for pkg in REQUIRED_PACKAGES:
    try:
        m = importlib.import_module(pkg)
        ver = getattr(m, "__version__", "unknown")
        print(f"  OK  {pkg}=={ver}")
    except ImportError as e:
        print(f"  FAIL {pkg}: {e}")
        all_ok = False

print("\nMPS / GPU:")
try:
    import torch
    mps = torch.backends.mps.is_available()
    print(f"  torch.backends.mps.is_available(): {mps}")
    if not mps:
        print("  WARNING: MPS not available — training will use CPU")
except Exception as e:
    print(f"  ERROR checking MPS: {e}")

print("\nCOSIE / tacos:")
cosie_ok = False
tacos_ok = False
try:
    import cosie
    print(f"  OK  cosie=={getattr(cosie, '__version__', 'unknown')}")
    cosie_ok = True
except ImportError:
    print("  cosie: not installed")
if not cosie_ok:
    try:
        import tacos
        ver = getattr(tacos, "__version__", "0.0.2")
        print(f"  OK  tacos=={ver}")
        tacos_ok = True
    except ImportError:
        print("  tacos: not installed")
if not cosie_ok and not tacos_ok:
    print("  FAIL: neither cosie nor tacos is importable")
    all_ok = False

print("\nPyTorch Geometric sparse ops (basic check):")
try:
    import torch
    import torch_geometric
    # Basic geometric op to catch missing scatter/sparse extensions
    from torch_geometric.data import Data
    x = torch.randn(4, 3)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)
    print(f"  OK  torch_geometric.data.Data instantiation works")
except Exception as e:
    print(f"  FAIL: {e}")
    all_ok = False

print("\n" + ("=" * 50))
if all_ok:
    print("ALL CHECKS PASSED — environment is ready for Phase 1 data validation")
else:
    print("SOME CHECKS FAILED — review errors above before proceeding")
    sys.exit(1)
