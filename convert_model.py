import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
LOCAL_PY = str(PROJECT_ROOT / "_tfjs_py")
if LOCAL_PY not in sys.path:
    sys.path.insert(0, LOCAL_PY)

MODEL_KERAS = PROJECT_ROOT / "model.keras"
MODEL_H5 = PROJECT_ROOT / "model.h5"
OUTPUT_DIR = PROJECT_ROOT / "public" / "model"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("==> Loading Keras model...")
import tensorflow as tf
from tensorflow import keras

loaded = False
last_err = None

for src, label in [(MODEL_KERAS, "model.keras"), (MODEL_H5, "model.h5")]:
    if not src.exists():
        print(f"  - Skipping {label} (not found)")
        continue
    try:
        print(f"  - Trying {label}: {src}")
        if label == "model.keras":
            try:
                model = keras.models.load_model(str(src), compile=False)
            except Exception as e1:
                print(f"    keras.models.load_model failed: {e1}")
                try:
                    model = tf.keras.models.load_model(str(src), compile=False)
                except Exception as e2:
                    print(f"    tf.keras.models.load_model also failed: {e2}")
                    raise
        else:
            model = keras.models.load_model(str(src), compile=False)
        loaded = True
        print(f"  -> Loaded {label} successfully")
        if hasattr(model, "summary"):
            try:
                model.summary(print_fn=lambda x: None)
                print(f"  -> Model input: {getattr(model, 'input_shape', None)}")
                print(f"  -> Model output: {getattr(model, 'output_shape', None)}")
                layers = getattr(model, "layers", [])
                print(f"  -> Num layers: {len(layers)}")
            except Exception:
                pass
        break
    except Exception as e:
        last_err = e
        print(f"  - Failed to load {label}: {type(e).__name__}: {e}")

if not loaded:
    raise RuntimeError(f"Could not load either model.keras or model.h5. Last error: {last_err}")

print("==> Converting to TensorFlow.js LayersModel format...")
print(f"  -> Output dir: {OUTPUT_DIR}")

import tensorflowjs as tfjs

try:
    tfjs.converters.save_keras_model(model, str(OUTPUT_DIR))
except TypeError:
    try:
        tfjs.converters.save_keras_model(model, str(OUTPUT_DIR), quantization_dtype_map=None)
    except Exception as e2:
        print(f"  -> save_keras_model failed with options: {e2}")
        raise

files = sorted(os.listdir(OUTPUT_DIR))
json_files = [f for f in files if f.endswith(".json")]
bin_files = [f for f in files if f.endswith(".bin")]
print("==> Conversion complete. Output files:")
for f in json_files + bin_files:
    size = (OUTPUT_DIR / f).stat().st_size
    print(f"  - {f}  ({size/1024/1024:.2f} MB)")
print(f"  TOTAL: {len(json_files)} .json, {len(bin_files)} .bin")
if not json_files or "model.json" not in json_files:
    raise RuntimeError("model.json was not produced. Conversion failed.")
print("SUCCESS: model.json + weights are ready in ./public/model/")
