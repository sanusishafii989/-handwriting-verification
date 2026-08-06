"""
Flatten the nested Functional model topology in model.h5 into a single flat
layer list that TF.js loadLayersModel can consume.

The original model has three nesting levels:
  siamese_xception_l1 (Functional)
    ├── input_a, input_b (InputLayer)
    ├── embedding_model (Functional)  ← called twice (input_a, input_b)
    │     ├── branch_input (InputLayer)
    │     ├── gray_to_rgb (Concatenate)
    │     ├── xception (Functional)   ← called once internally, but x2 via parent
    │     │     ├── input_layer (InputLayer)
    │     │     └── ... 132 layers ...
    │     ├── gap, embedding_bn, embedding_dropout, embedding_dense
    │     └── l2_normalize (Lambda → L2NormalizeLayer)
    ├── l1_distance (Lambda → L1DistanceLayer)
    └── similarity (Dense sigmoid)

TF.js cannot handle nested Functional models — the InputLayer inside each
nested model has empty inbound_nodes, causing "Graph disconnected" errors.

This script flattens the topology by:
1. Extracting all layers from nested models into one flat list
2. Setting each nested InputLayer's inbound_nodes to N empty entries
   (one per call of the parent model)
3. Duplicating each internal layer's inbound_nodes N times, incrementing
   the node index for each call
4. Stripping weight name prefixes to match the flattened layer names
"""

import json
import os
import copy
import sys
from pathlib import Path

import h5py
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).parent.resolve()
MODEL_H5 = PROJECT_ROOT / "model.h5"
OUTPUT_DIR = PROJECT_ROOT / "public" / "model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_JSON = OUTPUT_DIR / "model.json"
OUTPUT_BIN = OUTPUT_DIR / "group1-shard1of1.bin"

# ─── Keras 3 → Keras 2 cleanup helpers (from convert_model.py) ───────────────

LAMBDA_REPLACEMENTS = {
    "l2_normalize": "L2NormalizeLayer",
    "l1_distance": "L1DistanceLayer",
}


def _tensor_ref_from_keras3(obj):
    if not isinstance(obj, dict):
        return None
    if obj.get("class_name") != "__keras_tensor__":
        return None
    history = obj.get("config", {}).get("keras_history")
    if not history or len(history) < 3:
        return None
    return [history[0], history[1], history[2]]


def _convert_keras3_inbound_nodes(inbound_nodes):
    if not inbound_nodes:
        return inbound_nodes
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
    if isinstance(value, dict):
        if value.get("class_name") == "DTypePolicy":
            dtype_name = value.get("config", {}).get("name", "float32")
            if dtype_name == "mixed_float16":
                return "float32"
            return dtype_name
        return {k: _normalize_dtype(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_dtype(v) for v in value]
    return value


def _fix_regularizer(obj):
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
    if not refs or not isinstance(refs, list):
        return refs
    if refs and isinstance(refs[0], list):
        return refs
    if len(refs) == 3 and isinstance(refs[0], str):
        return [refs]
    return refs


def _fix_input_layer(layer_dict):
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
    if not isinstance(layer_dict, dict):
        return layer_dict
    class_name = layer_dict.get("class_name")
    if class_name == "Lambda":
        layer_name = layer_dict.get("name") or layer_dict.get("config", {}).get("name")
        if layer_name in LAMBDA_REPLACEMENTS:
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
        layer_dict["inbound_nodes"] = _convert_keras3_inbound_nodes(
            layer_dict["inbound_nodes"]
        )
    return layer_dict


def clean_keras3_to_keras2(obj):
    if isinstance(obj, dict):
        for k in [
            "module", "registered_name", "build_input_shape",
            "optional", "sparse", "ragged", "quantization_config", "autocast",
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


# ─── Flattening logic ────────────────────────────────────────────────────────

def flatten_model(topology):
    """
    Flatten nested Functional models into a single flat layer list.

    For a nested model called N times:
    - Its InputLayer gets N empty inbound_nodes entries: [[], [], ...]
    - Each internal layer gets N inbound_nodes entries (duplicated from the
      original single entry, with node indices incremented per call)
    - References to the removed Functional wrapper name are replaced with
      references to the wrapper's output layer (last layer in the nested list)
    """
    flat_layers = []
    # Map: functional_wrapper_name → output_layer_name
    wrapper_output_map = {}

    def get_output_layer_name(functional_layer):
        """Get the name of the last (output) layer inside a Functional wrapper."""
        nested_layers = functional_layer["config"]["layers"]
        output_layers = functional_layer["config"].get("output_layers", [])
        if output_layers and isinstance(output_layers[0], list) and len(output_layers[0]) > 0:
            return output_layers[0][0]
        # Fallback: last layer's name
        last = nested_layers[-1]
        return last.get("config", {}).get("name", last.get("name", ""))

    def process_layers(layers, n_calls):
        for layer in layers:
            cls = layer.get("class_name", "")

            if cls == "Functional":
                # Record the mapping: wrapper_name → output_layer_name
                wrapper_name = layer.get("config", {}).get("name", layer.get("name", ""))
                output_name = get_output_layer_name(layer)
                wrapper_output_map[wrapper_name] = output_name
                print(f"    [flatten] {wrapper_name} → output: {output_name}")

                # Determine this model's call count
                nested_ib = layer.get("inbound_nodes", [])
                nested_n = max(len(nested_ib), n_calls) if nested_ib else n_calls
                # Recurse into the nested model's layers
                process_layers(layer["config"]["layers"], nested_n)
                # Do NOT add the Functional wrapper itself

            elif cls == "InputLayer":
                # N empty entries for N calls
                layer["inbound_nodes"] = [[] for _ in range(n_calls)]
                flat_layers.append(layer)

            else:
                # Regular layer — ensure n_calls entries in inbound_nodes
                current_ib = layer.get("inbound_nodes", [])
                if len(current_ib) == 0:
                    flat_layers.append(layer)
                elif len(current_ib) >= n_calls:
                    flat_layers.append(layer)
                else:
                    # Duplicate the single entry n_calls times
                    base_entry = current_ib[0]
                    new_ib = []
                    for call_idx in range(n_calls):
                        entry = []
                        for ref in base_entry:
                            new_ref = list(ref)
                            new_ref[1] = ref[1] + call_idx
                            entry.append(new_ref)
                        new_ib.append(entry)
                    layer["inbound_nodes"] = new_ib
                    flat_layers.append(layer)

    process_layers(topology["config"]["layers"], 1)

    # Replace all references to removed Functional wrappers with their output layers
    print("  Replacing wrapper references:")
    for layer in flat_layers:
        ib = layer.get("inbound_nodes", [])
        for node_idx, node in enumerate(ib):
            for ref_idx, ref in enumerate(node):
                if isinstance(ref, list) and len(ref) > 0:
                    ref_name = ref[0]
                    if ref_name in wrapper_output_map:
                        old_name = ref[0]
                        ref[0] = wrapper_output_map[old_name]
                        # The node index stays the same — the output layer
                        # already has N inbound_nodes entries matching the
                        # wrapper's call count
                        print(f"    {layer['config']['name']}: [{old_name}, {ref[1]}] → [{ref[0]}, {ref[1]}]")

    topology["config"]["layers"] = flat_layers
    return topology


# ─── Weight collection (from convert_model.py) ──────────────────────────────

SKIP_GROUPS = {"top_level_model_weights"}


def collect_weights(group, path_prefix_parts):
    weights = []
    for name in group:
        obj = group[name]
        if name in SKIP_GROUPS:
            continue
        if isinstance(obj, h5py.Dataset):
            data = np.asarray(obj[...])
            name_parts = path_prefix_parts + [name]
            weight_name = "/".join(name_parts) + ":0"

            if data.dtype == np.float64:
                data = data.astype(np.float32)
            elif data.dtype == np.float16:
                data = data.astype(np.float32)
            elif data.dtype == np.int64:
                data = data.astype(np.int32)
            elif data.dtype == np.bool_:
                data = data.astype(np.int32)

            weights.append({
                "name": weight_name,
                "shape": list(data.shape),
                "dtype": data.dtype.name,
                "data": data,
            })
        elif isinstance(obj, h5py.Group):
            def _has_ds(g, depth=0):
                if depth > 8:
                    return False
                for nm, v in g.items():
                    if isinstance(v, h5py.Dataset):
                        return True
                    if isinstance(v, h5py.Group) and _has_ds(v, depth + 1):
                        return True
                return False
            if not _has_ds(obj):
                continue
            weights.extend(collect_weights(obj, path_prefix_parts + [name]))
    return weights


def strip_weight_name(full_name):
    """
    Strip parent-model prefixes from a weight name.

    'embedding_model/xception/block1_conv1/kernel:0' → 'block1_conv1/kernel:0'
    'similarity/similarity/bias:0'                     → 'similarity/bias:0'
    """
    parts = full_name.rsplit("/", 2)  # ['path', 'layer_name', 'weight:0']
    if len(parts) == 3:
        return parts[1] + "/" + parts[2]
    elif len(parts) == 2:
        return full_name
    return full_name


# ─── Main conversion ────────────────────────────────────────────────────────

print("=" * 60)
print("Flattening TF.js converter for nested Siamese model")
print("=" * 60)

# 1. Read and clean topology
print("\n[1/5] Reading Keras model topology from model.h5 ...")
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
print(f"  num layers cfg : {len(topology['config']['layers'])}")

topology = clean_keras3_to_keras2(topology)
topology["keras_version"] = "2.14.0"
topology["backend"] = "tensorflow"
print("  topology cleaned: Keras 3 extras stripped")

# 2. Flatten topology
print("\n[2/5] Flattening nested Functional models ...")
topology = flatten_model(topology)
n_flat = len(topology["config"]["layers"])
print(f"  Flat layer count: {n_flat}")

# Print layer summary
for i, layer in enumerate(topology["config"]["layers"]):
    cls = layer.get("class_name", "?")
    name = layer.get("config", {}).get("name", layer.get("name", "?"))
    ib = layer.get("inbound_nodes", [])
    print(f"  {i:3d}: {cls:30s}  name={name:30s}  inbound_nodes={len(ib)}")

# 3. Collect weights
print("\n[3/5] Collecting weight tensors from HDF5 ...")
mw = f["model_weights"]
all_weights = collect_weights(mw, [])
total_params = sum(int(np.prod(w["shape"])) if w["shape"] else 1 for w in all_weights)
total_bytes = sum(w["data"].nbytes for w in all_weights)
print(f"  -> {len(all_weights)} weight tensors, {total_params:,} params, {total_bytes/1024/1024:.1f} MB")

# 4. Strip weight names to match flattened topology
print("\n[4/5] Stripping weight name prefixes ...")
for w in all_weights:
    w["name"] = strip_weight_name(w["name"])
print(f"  Example stripped names:")
for w in all_weights[:3]:
    print(f"    {w['name']}  shape={w['shape']}")
for w in all_weights[-3:]:
    print(f"    {w['name']}  shape={w['shape']}")

f.close()

# 5. Write .bin and model.json
print(f"\n[5/5] Writing output files ...")
with open(str(OUTPUT_BIN), "wb") as binfile:
    for w in all_weights:
        arr_bytes = w["data"].astype(np.float32).tobytes(order="C")
        binfile.write(arr_bytes)

bin_size = OUTPUT_BIN.stat().st_size
print(f"  -> {OUTPUT_BIN.name}: {bin_size:,} bytes ({bin_size/1024/1024:.1f} MB)")

manifest_weights = []
for w in all_weights:
    manifest_weights.append({
        "name": w["name"],
        "shape": w["shape"],
        "dtype": "float32",
    })

weights_manifest = [{
    "paths": [OUTPUT_BIN.name],
    "weights": manifest_weights,
}]

model_json = {
    "modelTopology": topology,
    "weightsManifest": weights_manifest,
    "format": "layers-model",
    "generatedBy": "flatten-converter-1.0",
    "convertedBy": "project-local flatten converter",
    "sourceKerasVersion": keras_version,
    "sourceKerasBackend": backend,
}

with open(str(OUTPUT_JSON), "w", encoding="utf-8") as jf:
    json.dump(model_json, jf, indent=2, ensure_ascii=False)

json_size = OUTPUT_JSON.stat().st_size
print(f"  -> {OUTPUT_JSON.name}: {json_size:,} bytes ({json_size/1024:.1f} KB)")

# Verification
print("\n" + "=" * 60)
print("VERIFICATION")
print("=" * 60)
ok = OUTPUT_JSON.exists() and json_size > 1000 and OUTPUT_BIN.exists() and bin_size > 1_000_000
if ok:
    print("[OK] SUCCESS: Flattened TF.js LayersModel files are ready.")
    print(f"   - {OUTPUT_JSON}")
    print(f"   - {OUTPUT_BIN}")
else:
    print("[FAIL] Files were not produced correctly.")
    sys.exit(1)
