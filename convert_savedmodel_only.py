"""
Convert the already-created SavedModel to TF.js GraphModel.
Uses the local _tfjs_py tensorflowjs 4.14.0 with mocked optional deps.
"""
import sys
import os
import shutil
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).parent.resolve()
SAVED_MODEL_DIR = PROJECT_ROOT / "_saved_model_tmp"
OUTPUT_DIR = PROJECT_ROOT / "public" / "model"

if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Mock optional modules that _tfjs_py tensorflowjs 4.14.0 tries to import
import types
for mod_name in ['tensorflow_decision_forests', 'tensorflow_hub', 'jax', 'jax.experimental', 'jax.experimental.jax2tf']:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)
# jax.experimental.jax2tf needs shape_poly with PolyShape and Constraints
sp = types.ModuleType('shape_poly')
class _Dummy:
    def __init__(self, *a, **kw): pass
    def __call__(self, *a, **kw): return self
    def __getattr__(self, name): return _Dummy()
sp.PolyShape = _Dummy
sp.Constraints = _Dummy
sys.modules['jax.experimental.jax2tf'].shape_poly = sp

# Use local _tfjs_py first (before pip-installed version)
LOCAL_PY = str(PROJECT_ROOT / "_tfjs_py")
sys.path.insert(0, LOCAL_PY)

print("=" * 60)
print("Convert SavedModel -> TF.js GraphModel")
print("=" * 60)

import tensorflow as tf
print(f"TF: {tf.__version__}")
import keras
print(f"Keras: {keras.__version__}")

import tensorflowjs as tfjs
print(f"TFJS: {tfjs.__version__}")
print(f"TFJS path: {tfjs.__file__}")

print(f"\nSavedModel: {SAVED_MODEL_DIR}")
print(f"Output: {OUTPUT_DIR}")

print("\nConverting ...")
tfjs.converters.convert_tf_saved_model(
    str(SAVED_MODEL_DIR),
    str(OUTPUT_DIR),
)

# Clean up
if SAVED_MODEL_DIR.exists():
    shutil.rmtree(SAVED_MODEL_DIR)

# Verify output
files = sorted(os.listdir(OUTPUT_DIR))
json_files = [f for f in files if f.endswith(".json")]
bin_files = [f for f in files if f.endswith(".bin")]
print("\n" + "=" * 60)
print("OUTPUT FILES:")
for f in json_files + bin_files:
    size = (OUTPUT_DIR / f).stat().st_size
    print(f"  {f}  ({size/1024/1024:.2f} MB)")

if not json_files:
    raise RuntimeError("No model.json produced - conversion failed!")

print("\n[OK] SUCCESS: TF.js GraphModel files ready in public/model/")
