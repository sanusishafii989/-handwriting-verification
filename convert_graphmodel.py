"""
Rebuild the Siamese model architecture in Keras 3, load weights from model.h5,
export to SavedModel, then convert to TF.js GraphModel.
"""
import sys
import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
LOCAL_PY = str(PROJECT_ROOT / "_tfjs_py")
if LOCAL_PY not in sys.path:
    sys.path.insert(0, LOCAL_PY)

MODEL_H5 = PROJECT_ROOT / "model.h5"
OUTPUT_DIR = PROJECT_ROOT / "public" / "model"
SAVED_MODEL_DIR = PROJECT_ROOT / "_saved_model_tmp"

if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
if SAVED_MODEL_DIR.exists():
    shutil.rmtree(SAVED_MODEL_DIR)

print("=" * 60)
print("Rebuild + Convert to TF.js GraphModel")
print("=" * 60)

os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
import tensorflow as tf
import numpy as np
import h5py
import json

keras.config.enable_unsafe_deserialization()

# 1. Rebuild the embedding model architecture
print("\n[1/4] Rebuilding Siamese model architecture ...")

IMG_H, IMG_W, CHANNELS = 112, 896, 1

def build_embedding_model():
    inputs = keras.Input(shape=(IMG_H, IMG_W, CHANNELS), name="branch_input")
    x = keras.layers.Concatenate(axis=-1, name="gray_to_rgb")([inputs, inputs, inputs])

    backbone = keras.applications.Xception(
        include_top=False, weights="imagenet",
        input_shape=(IMG_H, IMG_W, 3), pooling=None
    )
    backbone._name = "xception_backbone"
    backbone.trainable = False
    x = backbone(x)
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = keras.layers.BatchNormalization(name="embedding_bn")(x)
    x = keras.layers.Dropout(0.35, name="embedding_dropout")(x)
    x = keras.layers.Dense(64, name="embedding_dense")(x)
    x = keras.layers.Lambda(lambda v: tf.math.l2_normalize(v, axis=-1), name="l2_normalize")(x)
    return keras.Model(inputs, x, name="embedding_model")

embedding_model = build_embedding_model()

input_a = keras.Input(shape=(IMG_H, IMG_W, CHANNELS), name="input_a")
input_b = keras.Input(shape=(IMG_H, IMG_W, CHANNELS), name="input_b")

embedding_a = embedding_model(input_a)
embedding_b = embedding_model(input_b)

distance = keras.layers.Lambda(lambda v: tf.abs(v[0] - v[1]), name="l1_distance")([embedding_a, embedding_b])
output = keras.layers.Dense(1, activation="sigmoid", name="similarity", dtype="float32")(distance)

siamese_model = keras.Model(inputs=[input_a, input_b], outputs=output, name="siamese_xception_l1")
print(f"  Input shapes:  {siamese_model.input_shape}")
print(f"  Output shape:  {siamese_model.output_shape}")

# 2. Load weights from H5 file
print("\n[2/4] Loading weights from model.h5 ...")
f = h5py.File(str(MODEL_H5), "r")
model_config_raw = f.attrs["model_config"]
if isinstance(model_config_raw, bytes):
    model_config_raw = model_config_raw.decode("utf-8")
original_config = json.loads(model_config_raw)

# The original model was saved by Keras 3 with mixed_float16 policy
# We need to extract weights from the H5 file and transfer them

# Get the list of weight tensors from the H5 file
def collect_weight_names(group, prefix=""):
    names = []
    for name in group:
        obj = group[name]
        if name == "top_level_model_weights":
            continue
        if isinstance(obj, h5py.Dataset):
            full = (prefix + "/" + name) if prefix else name
            names.append((full, obj))
        elif isinstance(obj, h5py.Group):
            sub_prefix = (prefix + "/" + name) if prefix else name
            names.extend(collect_weight_names(obj, sub_prefix))
    return names

# Load all weights into a dict by name
mw = f["model_weights"]
all_weight_data = {}
def walk_weights(group, path=""):
    for name in group:
        obj = group[name]
        if name == "top_level_model_weights":
            continue
        if isinstance(obj, h5py.Dataset):
            full = (path + "/" + name) if path else name
            all_weight_data[full] = np.asarray(obj[...])
        elif isinstance(obj, h5py.Group):
            sub_path = (path + "/" + name) if path else name
            walk_weights(obj, sub_path)

walk_weights(mw)
print(f"  Found {len(all_weight_data)} weight tensors in H5")

# Map weights to our rebuilt model layers
# The H5 stores weights as: embedding_model/xception_backbone/block1_conv1/kernel:0 etc.
# Our rebuilt model layers have the same names

def set_layer_weights(model, weight_data, prefix=""):
    for layer in model.layers:
        layer_name = layer.name
        # Check if this layer has weights
        layer_weights = []
        for wname, wdata in weight_data.items():
            # Match by layer name in the path
            parts = wname.split("/")
            if len(parts) >= 2:
                # Check if the second-to-last part matches layer name
                if parts[-2] == layer_name:
                    layer_weights.append((parts[-1], wdata))
                # Also check with prefix (for nested models)
                elif len(parts) >= 3 and parts[-3] == layer_name:
                    # Could be a nested model layer
                    pass

        if layer_weights:
            # Sort to get consistent order (bias before kernel, etc.)
            # Keras typically expects [kernel, bias] or [gamma, beta, moving_mean, moving_variance]
            weight_names = [w[0] for w in layer_weights]
            weight_values = [w[1] for w in layer_weights]

            # Get expected weight names from layer
            expected = layer.get_weights()
            if len(expected) != len(weight_values):
                print(f"    {layer_name}: weight count mismatch ({len(expected)} vs {len(weight_values)})")
                continue

            # Try to match by order (kernel, bias) or (gamma, beta, moving_mean, moving_variance)
            try:
                # Sort our weights to match expected order
                # Keras convention: kernel, bias OR gamma, beta, moving_mean, moving_variance
                order_map = {
                    "kernel": 0, "bias": 1,
                    "gamma": 0, "beta": 1, "moving_mean": 2, "moving_variance": 3,
                    "depthwise_kernel": 0,
                }
                indexed = []
                for wname, wdata in layer_weights:
                    wkey = wname.replace(":0", "")
                    idx = order_map.get(wkey, 99)
                    indexed.append((idx, wdata))
                indexed.sort(key=lambda x: x[0])
                sorted_values = [v for _, v in indexed]

                if len(sorted_values) == len(expected):
                    # Check shapes match
                    shapes_ok = all(s.shape == e.shape for s, e in zip(sorted_values, expected))
                    if shapes_ok:
                        layer.set_weights(sorted_values)
                    else:
                        print(f"    {layer_name}: shape mismatch")
                else:
                    print(f"    {layer_name}: count mismatch after sort ({len(sorted_values)} vs {len(expected)})")
            except Exception as e:
                print(f"    {layer_name}: error setting weights: {e}")

# Actually, let's use a simpler approach: load by name recursively
print("  Setting weights by layer name ...")

def find_weights_for_layer(layer_name, weight_data):
    """Find all weights belonging to a layer by name."""
    results = []
    for wname, wdata in weight_data.items():
        parts = wname.split("/")
        if len(parts) >= 2 and parts[-2] == layer_name:
            results.append((parts[-1].replace(":0", ""), wdata))
    return results

def transfer_weights(model, weight_data, depth=0):
    prefix = "  " * depth
    for layer in model.layers:
        if hasattr(layer, 'layers') and layer.layers:
            # Nested model (embedding_model, xception_backbone)
            print(f"{prefix}{layer.name}: (nested model, recursing)")
            transfer_weights(layer, weight_data, depth + 1)
        else:
            wts = find_weights_for_layer(layer.name, weight_data)
            if wts:
                expected = layer.get_weights()
                if len(wts) == len(expected):
                    # Sort by convention
                    order_map = {
                        "kernel": 0, "bias": 1, "gamma": 0, "beta": 1,
                        "moving_mean": 2, "moving_variance": 3, "depthwise_kernel": 0,
                    }
                    indexed = [(order_map.get(n, 99), v) for n, v in wts]
                    indexed.sort(key=lambda x: x[0])
                    values = [v for _, v in indexed]
                    if all(v.shape == e.shape for v, e in zip(values, expected)):
                        layer.set_weights(values)
                        print(f"{prefix}{layer.name}: set {len(values)} weights OK")
                    else:
                        print(f"{prefix}{layer.name}: shape mismatch")
                else:
                    print(f"{prefix}{layer.name}: count mismatch ({len(wts)} vs {len(expected)})")

transfer_weights(siamese_model, all_weight_data)
f.close()

# 3. Export to SavedModel
print("\n[3/4] Exporting to TensorFlow SavedModel ...")

@tf.function(input_signature=[
    tf.TensorSpec([None, 112, 896, 1], tf.float32, name="input_a"),
    tf.TensorSpec([None, 112, 896, 1], tf.float32, name="input_b"),
])
def serving_fn(input_a, input_b):
    output = siamese_model([input_a, input_b])
    return {"similarity": output}

tf.saved_model.save(
    siamese_model,
    str(SAVED_MODEL_DIR),
    signatures={"serving_default": serving_fn.get_concrete_function()},
)
print(f"  -> Saved to {SAVED_MODEL_DIR}")

# 4. Convert to TF.js GraphModel
print("\n[4/4] Converting to TF.js GraphModel ...")
import tensorflowjs as tfjs

tfjs.converters.convert_tf_saved_model(
    str(SAVED_MODEL_DIR),
    str(OUTPUT_DIR),
)

# Clean up
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
    raise RuntimeError("No model.json produced — conversion failed!")

print("\n[OK] SUCCESS: TF.js GraphModel files ready in public/model/")
