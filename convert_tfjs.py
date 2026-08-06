import json
import os
import struct
import sys
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).parent.resolve()
MODEL_H5 = PROJECT_ROOT / "model.h5"
OUTPUT_DIR = PROJECT_ROOT / "public" / "model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_JSON = OUTPUT_DIR / "model.json"
OUTPUT_BIN = OUTPUT_DIR / "group1-shard1of1.bin"

print("=" * 60)
print("Handwriting Siamese Model -> TensorFlow.js LayersModel converter")
print("=" * 60)
print(f"Input : {MODEL_H5} ({MODEL_H5.stat().st_size / 1024 / 1024:.1f} MB)")
print(f"Output: {OUTPUT_DIR}/")

# ---------------- 1. Load Keras topology from model.h5 root attrs ----------------
print("\n[1/4] Reading Keras model topology (model.h5 attrs) ...")
f = h5py.File(str(MODEL_H5), "r")

model_config_raw = f.attrs["model_config"]
if isinstance(model_config_raw, bytes):
    model_config_raw = model_config_raw.decode("utf-8")
topology = json.loads(model_config_raw)

keras_version = f.attrs.get("keras_version", "")
if isinstance(keras_version, bytes):
    keras_version = keras_version.decode("utf-8")
backend = f.attrs.get("backend", "")
if isinstance(backend, bytes):
    backend = backend.decode("utf-8")

print(f"  class_name     : {topology.get('class_name')}")
print(f"  keras_version  : {keras_version}")
print(f"  backend        : {backend}")
print(f"  num layers cfg : {len(topology['config']['layers'])}")

# ---------------- 2. Strip Keras 3 extras and convert topology for TF.js ----------------

LAMBDA_REPLACEMENTS = {
    "l2_normalize": "L2NormalizeLayer",
    "l1_distance": "L1DistanceLayer",
}


def _tensor_ref_from_keras3(obj):
    """Extract [layer, node, tensor] from a Keras 3 __keras_tensor__ descriptor."""
    if not isinstance(obj, dict):
        return None
    if obj.get("class_name") != "__keras_tensor__":
        return None
    history = obj.get("config", {}).get("keras_history")
    if not history or len(history) < 3:
        return None
    return [history[0], history[1], history[2]]


def _convert_keras3_inbound_nodes(inbound_nodes):
    """Convert Keras 3 {args, kwargs} inbound nodes to TF.js/Keras 2 format."""
    if not inbound_nodes:
        return inbound_nodes

    # Already Keras 2: [[["layer", 0, 0, {}], ...], ...]
    if isinstance(inbound_nodes[0], list):
        return inbound_nodes

    converted = []
    for node in inbound_nodes:
        if isinstance(node, list):
            converted.append(node)
            continue
        if not isinstance(node, dict) or "args" not in node:
            converted.append(node)
            continue

        kwargs = node.get("kwargs") or {}
        node_args = node.get("args") or []
        refs = []

        for arg in node_args:
            if isinstance(arg, list):
                for item in arg:
                    ref = _tensor_ref_from_keras3(item)
                    if ref is not None:
                        refs.append(ref)
            else:
                ref = _tensor_ref_from_keras3(arg)
                if ref is not None:
                    refs.append(ref)

        converted.append([ref + [kwargs] for ref in refs])

    return converted


def _normalize_dtype(value):
    """Flatten Keras 3 DTypePolicy objects to plain dtype strings for TF.js."""
    if isinstance(value, dict):
        if value.get("class_name") == "DTypePolicy":
            dtype_name = value.get("config", {}).get("name", "float32")
            # TF.js stores float32 weights; mixed_float16 topology still runs in float32.
            if dtype_name == "mixed_float16":
                return "float32"
            return dtype_name
        return {k: _normalize_dtype(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_dtype(v) for v in value]
    return value


def _fix_regularizer(obj):
    """Map Keras 3 regularizer names to TF.js-compatible serializers."""
    if not isinstance(obj, dict) or "class_name" not in obj:
        return obj

    class_name = obj.get("class_name")
    cfg = obj.get("config", {}) or {}

    if class_name == "L2":
        return {"class_name": "L1L2", "config": {"l1": 0, "l2": cfg.get("l2", 0)}}
    if class_name == "L1":
        return {"class_name": "L1L2", "config": {"l1": cfg.get("l1", 0), "l2": 0}}

    return obj


def _normalize_layer_refs(refs):
    """Convert flat [name, node, idx] refs to TF.js [[name, node, idx], ...] format."""
    if not refs or not isinstance(refs, list):
        return refs
    if refs and isinstance(refs[0], list):
        return refs
    if len(refs) == 3 and isinstance(refs[0], str):
        return [refs]
    return refs


def _fix_input_layer(layer_dict):
    """Convert Keras 3 InputLayer batch_shape to TF.js batchInputShape."""
    if not isinstance(layer_dict, dict):
        return layer_dict
    if layer_dict.get("class_name") != "InputLayer":
        return layer_dict

    cfg = layer_dict.get("config", {})
    if "batch_shape" in cfg and "batchInputShape" not in cfg:
        cfg["batchInputShape"] = cfg.pop("batch_shape")
    elif "batch_shape" in cfg:
        cfg.pop("batch_shape", None)

    layer_dict["config"] = cfg
    return layer_dict


def _replace_lambda_layers(layer_dict):
    """Replace unsupported Python Lambda layers with TF.js custom layer class names."""
    if not isinstance(layer_dict, dict):
        return layer_dict

    class_name = layer_dict.get("class_name")
    layer_name = layer_dict.get("name") or layer_dict.get("config", {}).get("name")

    if class_name == "Lambda" and layer_name in LAMBDA_REPLACEMENTS:
        replacement = LAMBDA_REPLACEMENTS[layer_name]
        layer_dict["class_name"] = replacement
        cfg = layer_dict.get("config", {})
        layer_dict["config"] = {
            "name": cfg.get("name", layer_name),
            "trainable": cfg.get("trainable", True),
            "dtype": _normalize_dtype(cfg.get("dtype", "float32")),
        }

    layer_dict = _fix_input_layer(layer_dict)

    if "config" in layer_dict and isinstance(layer_dict["config"], dict):
        cfg = layer_dict["config"]
        if "layers" in cfg and isinstance(cfg["layers"], list):
            cfg["layers"] = [_replace_lambda_layers(layer) for layer in cfg["layers"]]
        layer_dict["config"] = cfg

    if "inbound_nodes" in layer_dict:
        layer_dict["inbound_nodes"] = _convert_keras3_inbound_nodes(layer_dict["inbound_nodes"])

    return layer_dict


def clean_keras3_to_keras2(obj):
    """Recursively convert Keras 3 topology to TF.js-compatible Keras 2 format."""
    if isinstance(obj, dict):
        for k in [
            "module",
            "registered_name",
            "build_input_shape",
            "optional",
            "sparse",
            "ragged",
            "quantization_config",
            "autocast",
        ]:
            obj.pop(k, None)

        if obj.get("class_name") == "DTypePolicy":
            return _normalize_dtype(obj)

        if obj.get("class_name") in ("L1", "L2"):
            return _fix_regularizer(obj)

        if "class_name" in obj and "config" in obj and isinstance(obj["config"], dict):
            if "name" not in obj and "name" in obj["config"]:
                obj["name"] = obj["config"]["name"]
            obj = _replace_lambda_layers(obj)
            obj["config"] = clean_keras3_to_keras2(obj["config"])
            if "dtype" in obj["config"]:
                obj["config"]["dtype"] = _normalize_dtype(obj["config"]["dtype"])
            return obj

        cleaned = {}
        for k, v in obj.items():
            if k == "inbound_nodes":
                cleaned[k] = _convert_keras3_inbound_nodes(v)
            elif k in ("input_layers", "output_layers"):
                cleaned[k] = _normalize_layer_refs(v)
            elif k.endswith("_regularizer"):
                cleaned[k] = _fix_regularizer(v) if isinstance(v, dict) else v
            elif k == "dtype":
                cleaned[k] = _normalize_dtype(v)
            else:
                cleaned[k] = clean_keras3_to_keras2(v)
        return cleaned

    if isinstance(obj, list):
        return [clean_keras3_to_keras2(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(clean_keras3_to_keras2(x) for x in obj)
    return obj


topology_clean = clean_keras3_to_keras2(topology)
topology_clean["keras_version"] = "2.14.0"  # Tell TF.js this is Keras v2-compatible
topology_clean["backend"] = "tensorflow"
print("  topology cleaned: Keras 3 extras stripped, reported version -> 2.14.0")

# ---------------- 3. Read weights using recursive HDF5 group walk ----------------
print("\n[2/4] Collecting weight tensors via recursive HDF5 walk of model_weights/ ...")
mw = f["model_weights"]

all_weights = []  # list of dicts: {name, shape, dtype, data (numpy)}
total_params = 0
total_bytes = 0

SKIP_GROUPS = {"top_level_model_weights"}  # metadata-only, no actual weights

def collect_weights(group, path_prefix_parts):
    """Recursively walk HDF5 group and collect Dataset objects as weights.
    - group: current h5py.Group being walked
    - path_prefix_parts: list of ancestor group names forming the TF.js weight scope
    """
    global total_params, total_bytes, all_weights
    for name in group:
        obj = group[name]
        if name in SKIP_GROUPS:
            continue
        if isinstance(obj, h5py.Dataset):
            # We hit a weight tensor!
            data = np.asarray(obj[...])
            # Build canonical weight name: path/to/weight:0
            name_parts = path_prefix_parts + [name]
            weight_name = "/".join(name_parts) + ":0"

            # Cast for TF.js compatibility: everything standard dtypes
            cast_to = None
            if data.dtype == np.float64:
                cast_to = np.float32
            elif data.dtype == np.float16:
                cast_to = np.float32
            elif data.dtype == np.int64:
                cast_to = np.int32
            elif data.dtype == np.bool_:
                cast_to = np.int32
            if cast_to is not None:
                data = data.astype(cast_to)

            n_params = int(np.prod(data.shape)) if data.shape else 1
            n_bytes = data.nbytes
            total_params += n_params
            total_bytes += n_bytes
            all_weights.append({
                "name": weight_name,
                "shape": list(data.shape),
                "dtype": data.dtype.name,
                "data": data,
            })
        elif isinstance(obj, h5py.Group):
            # Skip empty leaf groups (like input_a/, input_b/, l1_distance/ that have weight_names=[])
            sub_has_datasets = False
            def _has_ds(g, depth=0):
                if depth > 8: return False
                for nm, v in g.items():
                    if isinstance(v, h5py.Dataset): return True
                    if isinstance(v, h5py.Group):
                        if _has_ds(v, depth+1): return True
                return False
            if not _has_ds(obj):
                continue
            collect_weights(obj, path_prefix_parts + [name])

collect_weights(mw, [])
print(f"  -> Walked HDF5 tree and collected {len(all_weights)} weight tensors")
print(f"  -> Total parameters: {total_params:,}")
print(f"  -> Total bytes:      {total_bytes:,} ({total_bytes/1024/1024:.1f} MB)")

# Print summary of first 5 and last 5 weights for sanity
if len(all_weights) >= 10:
    print(f"\n  First 5 weights:")
    for w in all_weights[:5]:
        print(f"    {w['name']:75s} shape={str(w['shape']):20s} dtype={w['dtype']}")
    print(f"  Last 5 weights:")
    for w in all_weights[-5:]:
        print(f"    {w['name']:75s} shape={str(w['shape']):20s} dtype={w['dtype']}")
else:
    print("\n  All collected weights:")
    for w in all_weights:
        print(f"    {w['name']:75s} shape={str(w['shape']):20s} dtype={w['dtype']}")

f.close()

# ---------------- 4. Write single shard .bin ----------------
print(f"\n[3/4] Writing weight shard: {OUTPUT_BIN.name} ...")
with open(str(OUTPUT_BIN), "wb") as binfile:
    for w in all_weights:
        arr_bytes = w["data"].astype(
            np.float32 if w["dtype"] == "float32" else
            np.float64 if w["dtype"] == "float64" else
            np.int32 if w["dtype"] == "int32" else
            np.int16 if w["dtype"] == "int16" else
            np.int8 if w["dtype"] == "int8" else
            np.uint8 if w["dtype"] == "uint8" else
            np.float32
        ).tobytes(order="C")
        binfile.write(arr_bytes)

bin_size = OUTPUT_BIN.stat().st_size
print(f"  -> Wrote {bin_size:,} bytes ({bin_size/1024/1024:.1f} MB)")

# Build weightsManifest entries with byte offsets
manifest_weights = []
offset = 0
for w in all_weights:
    # Compute size of each weight in the shard
    dtype = w["dtype"]
    if dtype == "float32":
        item_size = 4
    elif dtype == "float64":
        item_size = 8
    elif dtype == "int32":
        item_size = 4
    elif dtype == "int16":
        item_size = 2
    elif dtype == "int8" or dtype == "uint8":
        item_size = 1
    elif dtype == "bool":
        item_size = 1
    else:
        # Unknown dtype: cast to float32
        item_size = 4
        dtype = "float32"
    n_params = int(np.prod(w["shape"])) if w["shape"] else 1
    w_bytes = n_params * item_size
    manifest_weights.append({
        "name": w["name"],
        "shape": w["shape"],
        "dtype": dtype,
    })
    offset += w_bytes

weights_manifest = [{
    "paths": [OUTPUT_BIN.name],
    "weights": manifest_weights,
}]

# ---------------- 5. Assemble and write model.json ----------------
print(f"\n[4/4] Writing model.json with topology + weights manifest ...")
model_json = {
    "modelTopology": topology_clean,
    "weightsManifest": weights_manifest,
    "format": "layers-model",
    "generatedBy": "handwritten-tfjs-converter-1.0",
    "convertedBy": "project-local pure-python (numpy+h5py)",
    "sourceKerasVersion": keras_version,
    "sourceKerasBackend": backend,
}

with open(str(OUTPUT_JSON), "w", encoding="utf-8") as jf:
    json.dump(model_json, jf, indent=2, ensure_ascii=False)

json_size = OUTPUT_JSON.stat().st_size
print(f"  -> model.json size: {json_size:,} bytes ({json_size/1024:.1f} KB)")
print(f"  -> Weight shards  : 1 x {OUTPUT_BIN.name} ({bin_size/1024/1024:.1f} MB)")
print(f"  -> Weights total   : {len(all_weights)} tensors, {total_params:,} params")

# ---------------- 6. Final verification ----------------
print("\n" + "=" * 60)
print("VERIFICATION")
print("=" * 60)
ok_json = OUTPUT_JSON.exists() and json_size > 1000
ok_bin = OUTPUT_BIN.exists() and bin_size > 1_000_000
if ok_json and ok_bin:
    print("[OK] SUCCESS: TensorFlow.js LayersModel files are ready.")
    print(f"   - {OUTPUT_JSON}")
    print(f"   - {OUTPUT_BIN}")
    print("\nTo use: browse to http://localhost:3001/verify and the app will")
    print("auto-load the model on page visit. Model Status card should show")
    print('"Model Ready" (green) instead of "Load Failed" (red).')
else:
    print("[FAIL] FAILED: Files were not produced correctly.")
    sys.exit(1)
