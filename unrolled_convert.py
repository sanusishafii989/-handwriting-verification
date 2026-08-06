"""
Unrolled TF.js converter for the Siamese model.

Instead of flattening with shared layers (duplicated inbound_nodes which cause
TF.js fromConfig timeouts), this script DUPLICATES all shared layers into
separate instances for each branch. The result is a flat, non-shared model
that TF.js can deserialize quickly.

Structure after unrolling:
  input_a → gray_to_rgb_a → [xception layers _a] → gap_a → embedding_bn_a →
            embedding_dropout_a → embedding_dense_a → l2_normalize_a
  input_b → gray_to_rgb_b → [xception layers _b] → gap_b → embedding_bn_b →
            embedding_dropout_b → embedding_dense_b → l2_normalize_b
  l2_normalize_a + l2_normalize_b → l1_distance → similarity

Weight names are mapped so each duplicated layer gets its own weight tensors
copied from the same source.
"""
import json
import os
import sys
from pathlib import Path
from collections import OrderedDict

import h5py
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).parent.resolve()
MODEL_H5 = PROJECT_ROOT / "model.h5"
OUTPUT_DIR = PROJECT_ROOT / "public" / "model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_JSON = OUTPUT_DIR / "model.json"
OUTPUT_BIN = OUTPUT_DIR / "group1-shard1of1.bin"

# ─── Keras 3 → Keras 2 cleanup (reused from convert_model.py) ───────────────

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


# ─── Weight collection ───────────────────────────────────────────────────────

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
    """Strip parent-model prefixes, keeping only layer_name/weight:0"""
    parts = full_name.rsplit("/", 2)
    if len(parts) == 3:
        return parts[1] + "/" + parts[2]
    elif len(parts) == 2:
        return full_name
    return full_name


# ─── Unrolling logic ─────────────────────────────────────────────────────────

def deep_copy_layer(layer):
    """Deep copy a layer config, renaming it with a suffix."""
    return json.loads(json.dumps(layer))


def rename_layer(layer, suffix):
    """Rename a layer's name and config.name with the given suffix."""
    layer = deep_copy_layer(layer)
    old_name = layer.get("name", "")
    new_name = old_name + suffix
    layer["name"] = new_name
    if "config" in layer:
        layer["config"]["name"] = new_name
    return layer, old_name, new_name


def update_inbound_refs(layer, name_map):
    """Update all inbound_nodes references using the name_map."""
    ib = layer.get("inbound_nodes", [])
    for node in ib:
        for ref in node:
            if isinstance(ref, list) and len(ref) > 0:
                old = ref[0]
                if old in name_map:
                    ref[0] = name_map[old]
    return layer


def unroll_model(topology):
    """
    Unroll the Siamese model by duplicating all shared layers.

    The embedding_model is called twice (input_a and input_b). We create
    two separate copies of all embedding layers with _a and _b suffixes.
    """
    # Step 1: Extract the top-level layers
    top_layers = topology["config"]["layers"]

    # Find the embedding_model Functional layer
    embedding_layer = None
    other_top_layers = []
    for layer in top_layers:
        name = layer.get("config", {}).get("name", layer.get("name", ""))
        if name == "embedding_model" and layer.get("class_name") == "Functional":
            embedding_layer = layer
        else:
            other_top_layers.append(layer)

    if embedding_layer is None:
        raise RuntimeError("embedding_model not found in topology")

    # Step 2: Get embedding model's internal layers (already cleaned)
    emb_layers = embedding_layer["config"]["layers"]

    # Get the output layer name of the embedding model
    emb_output_layers = embedding_layer["config"].get("output_layers", [])
    if emb_output_layers and isinstance(emb_output_layers[0], list):
        emb_output_name = emb_output_layers[0][0]
    else:
        emb_output_name = emb_layers[-1].get("config", {}).get("name", "")

    print(f"  Embedding model output: {emb_output_name}")
    print(f"  Embedding model has {len(emb_layers)} internal layers")

    # Step 3: Create two copies (branch A and branch B)
    # We need to recursively flatten the xception Functional inside embedding too
    def flatten_nested_functional(layers):
        """Recursively flatten any nested Functional models into a flat list."""
        flat = []
        for layer in layers:
            cls = layer.get("class_name", "")
            if cls == "Functional":
                # Get the output layer name
                out_layers = layer["config"].get("output_layers", [])
                if out_layers and isinstance(out_layers[0], list):
                    out_name = out_layers[0][0]
                else:
                    nested = layer["config"]["layers"]
                    out_name = nested[-1].get("config", {}).get("name", "")

                wrapper_name = layer.get("config", {}).get("name", layer.get("name", ""))
                # Flatten the nested layers
                nested_flat = flatten_nested_functional(layer["config"]["layers"])

                # Fix InputLayer: give it a single empty inbound_node
                for nl in nested_flat:
                    if nl.get("class_name") == "InputLayer":
                        nl["inbound_nodes"] = [[]]

                # Replace references to the wrapper name with the output layer name
                for nl in nested_flat:
                    ib = nl.get("inbound_nodes", [])
                    for node in ib:
                        for ref in node:
                            if isinstance(ref, list) and len(ref) > 0:
                                if ref[0] == wrapper_name:
                                    ref[0] = out_name

                flat.extend(nested_flat)
                # Store the mapping for parent layers
                if not hasattr(flatten_nested_functional, '_mappings'):
                    flatten_nested_functional._mappings = {}
                flatten_nested_functional._mappings[wrapper_name] = out_name
            else:
                flat.append(layer)
        return flat

    # Flatten the embedding model's layers (removes xception Functional wrapper)
    flat_emb_layers = flatten_nested_functional(emb_layers)
    print(f"  Flattened embedding: {len(flat_emb_layers)} layers")

    # Fix references to removed Functional wrappers
    wrapper_mappings = getattr(flatten_nested_functional, '_mappings', {})
    for layer in flat_emb_layers:
        ib = layer.get("inbound_nodes", [])
        for node in ib:
            for ref in node:
                if isinstance(ref, list) and len(ref) > 0:
                    if ref[0] in wrapper_mappings:
                        ref[0] = wrapper_mappings[ref[0]]

    # Also fix InputLayer inbound_nodes for branch_input
    for layer in flat_emb_layers:
        if layer.get("class_name") == "InputLayer":
            layer["inbound_nodes"] = [[]]  # Single call

    # Fix all non-InputLayer layers to have single inbound_node entries
    for layer in flat_emb_layers:
        if layer.get("class_name") != "InputLayer":
            ib = layer.get("inbound_nodes", [])
            if len(ib) > 1:
                layer["inbound_nodes"] = [ib[0]]  # Keep only first call's node

    # Create branch A copy
    branch_a_layers = []
    name_map_a = {}  # old_name → new_name_a
    for layer in flat_emb_layers:
        new_layer = deep_copy_layer(layer)
        old_name = new_layer.get("name", "")
        new_name = old_name + "_a"
        new_layer["name"] = new_name
        if "config" in new_layer:
            new_layer["config"]["name"] = new_name
        name_map_a[old_name] = new_name
        # Update inbound_nodes
        ib = new_layer.get("inbound_nodes", [])
        for node in ib:
            for ref in node:
                if isinstance(ref, list) and len(ref) > 0:
                    if ref[0] in name_map_a:
                        ref[0] = name_map_a[ref[0]]
        branch_a_layers.append(new_layer)

    # Create branch B copy
    branch_b_layers = []
    name_map_b = {}
    for layer in flat_emb_layers:
        new_layer = deep_copy_layer(layer)
        old_name = new_layer.get("name", "")
        new_name = old_name + "_b"
        new_layer["name"] = new_name
        if "config" in new_layer:
            new_layer["config"]["name"] = new_name
        name_map_b[old_name] = new_name
        # Update inbound_nodes
        ib = new_layer.get("inbound_nodes", [])
        for node in ib:
            for ref in node:
                if isinstance(ref, list) and len(ref) > 0:
                    if ref[0] in name_map_b:
                        ref[0] = name_map_b[ref[0]]
        branch_b_layers.append(new_layer)

    # Step 4: Build the final flat layer list
    final_layers = []

    # input_a InputLayer
    for layer in other_top_layers:
        name = layer.get("config", {}).get("name", layer.get("name", ""))
        if name == "input_a":
            layer_copy = deep_copy_layer(layer)
            layer_copy["inbound_nodes"] = [[]]
            final_layers.append(layer_copy)
            break

    # input_b InputLayer
    for layer in other_top_layers:
        name = layer.get("config", {}).get("name", layer.get("name", ""))
        if name == "input_b":
            layer_copy = deep_copy_layer(layer)
            layer_copy["inbound_nodes"] = [[]]
            final_layers.append(layer_copy)
            break

    # Branch A layers — connect branch_input_a to input_a
    for layer in branch_a_layers:
        name = layer.get("name", "")
        if "branch_input" in name:
            # Connect to input_a
            layer["inbound_nodes"] = [[["input_a", 0, 0, {}]]]
        final_layers.append(layer)

    # Branch B layers — connect branch_input_b to input_b
    for layer in branch_b_layers:
        name = layer.get("name", "")
        if "branch_input" in name:
            # Connect to input_b
            layer["inbound_nodes"] = [[["input_b", 0, 0, {}]]]
        final_layers.append(layer)

    # l1_distance — connects to l2_normalize_a and l2_normalize_b
    emb_output_a = emb_output_name + "_a"
    emb_output_b = emb_output_name + "_b"
    for layer in other_top_layers:
        name = layer.get("config", {}).get("name", layer.get("name", ""))
        if name == "l1_distance":
            layer_copy = deep_copy_layer(layer)
            layer_copy["inbound_nodes"] = [[
                [emb_output_a, 0, 0, {}],
                [emb_output_b, 0, 0, {}],
            ]]
            final_layers.append(layer_copy)
            break

    # similarity Dense — connects to l1_distance
    for layer in other_top_layers:
        name = layer.get("config", {}).get("name", layer.get("name", ""))
        if name == "similarity":
            layer_copy = deep_copy_layer(layer)
            layer_copy["inbound_nodes"] = [[["l1_distance", 0, 0, {}]]]
            final_layers.append(layer_copy)
            break

    topology["config"]["layers"] = final_layers
    topology["config"]["input_layers"] = [["input_a", 0, 0], ["input_b", 0, 0]]
    topology["config"]["output_layers"] = [["similarity", 0, 0]]

    return topology, name_map_a, name_map_b


# ─── Main ────────────────────────────────────────────────────────────────────

print("=" * 60)
print("Unrolled TF.js converter for Siamese model")
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
print("  topology cleaned")

# 2. Unroll
print("\n[2/5] Unrolling Siamese model (duplicating shared layers) ...")
topology, name_map_a, name_map_b = unroll_model(topology)
n_layers = len(topology["config"]["layers"])
print(f"  Unrolled layer count: {n_layers}")

# Print layer summary
for i, layer in enumerate(topology["config"]["layers"]):
    cls = layer.get("class_name", "?")
    name = layer.get("config", {}).get("name", layer.get("name", "?"))
    ib = layer.get("inbound_nodes", [])
    ib_str = str(len(ib)) if ib else "0"
    print(f"  {i:3d}: {cls:30s}  name={name:35s}  ib={ib_str}")

# 3. Collect weights
print("\n[3/5] Collecting weight tensors from HDF5 ...")
mw = f["model_weights"]
all_weights = collect_weights(mw, [])
print(f"  -> {len(all_weights)} source weight tensors")

# 4. Create duplicated weights for each branch
print("\n[4/5] Duplicating weights for unrolled branches ...")
final_weights = []

# Map: stripped_name → weight data
weight_map = OrderedDict()
for w in all_weights:
    stripped = strip_weight_name(w["name"])
    weight_map[stripped] = w

# For each branch, create weights with _a/_b suffixes
for stripped_name, w in weight_map.items():
    # Branch A
    final_weights.append({
        "name": stripped_name.replace("block", "block_a", 0) if False else None,
        "shape": w["shape"],
        "dtype": "float32",
        "data": w["data"],
    })

# Actually, let's be smarter: strip the name, then for branch A add _a suffix
# to the layer name portion, for branch B add _b suffix
final_weights = []
for stripped_name, w in weight_map.items():
    # stripped_name is like "block1_conv1/kernel:0" or "embedding_dense/bias:0"
    parts = stripped_name.rsplit("/", 1)
    if len(parts) == 2:
        layer_name, weight_part = parts
    else:
        layer_name, weight_part = "", stripped_name

    # Branch A
    final_weights.append({
        "name": f"{layer_name}_a/{weight_part}",
        "shape": w["shape"],
        "dtype": "float32",
        "data": w["data"].copy() if hasattr(w["data"], "copy") else np.array(w["data"]),
    })

    # Branch B (only for embedding model layers, not for l1_distance/similarity)
    if stripped_name.startswith("similarity/"):
        # similarity is not duplicated — just keep as is
        final_weights[-1]["name"] = stripped_name  # override back
        # Don't add branch B copy
    elif stripped_name.startswith("l1_distance/"):
        # l1_distance has no weights
        pass
    else:
        final_weights.append({
            "name": f"{layer_name}_b/{weight_part}",
            "shape": w["shape"],
            "dtype": "float32",
            "data": w["data"].copy() if hasattr(w["data"], "copy") else np.array(w["data"]),
        })

total_params = sum(int(np.prod(w["shape"])) if w["shape"] else 1 for w in final_weights)
print(f"  -> {len(final_weights)} final weight tensors, {total_params:,} params")

# Print sample weight names
for w in final_weights[:5]:
    print(f"    {w['name']}  shape={w['shape']}")
print("    ...")
for w in final_weights[-5:]:
    print(f"    {w['name']}  shape={w['shape']}")

f.close()

# 5. Write output
print(f"\n[5/5] Writing output files ...")
with open(str(OUTPUT_BIN), "wb") as binfile:
    for w in final_weights:
        arr_bytes = w["data"].astype(np.float32).tobytes(order="C")
        binfile.write(arr_bytes)

bin_size = OUTPUT_BIN.stat().st_size
print(f"  -> {OUTPUT_BIN.name}: {bin_size:,} bytes ({bin_size/1024/1024:.1f} MB)")

manifest_weights = []
for w in final_weights:
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
    "generatedBy": "unrolled-converter-1.0",
    "convertedBy": "project-local unrolled converter",
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
    print("[OK] SUCCESS: Unrolled TF.js LayersModel files are ready.")
    print(f"   - {OUTPUT_JSON}")
    print(f"   - {OUTPUT_BIN}")
else:
    print("[FAIL] Files were not produced correctly.")
    sys.exit(1)
