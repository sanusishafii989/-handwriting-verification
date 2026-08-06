

## Codely Structured Memories

### User

### Feedback
- [2026-08-03 18:39:16] Environment issue: NODE_ENV is globally set to "production" on this machine. This causes `npm install` to skip devDependencies (tailwindcss, autoprefixer, etc.). Must run `$env:NODE_ENV="development"; npm install` to get all packages. **Why:** Windows environment variable set outside the project. **How to apply:** Always set NODE_ENV=development before npm install or npm run dev.

### Project
- [2026-08-03 18:39:16] Project: Handwriting verification system — Siamese Xception network (112×896 grayscale input, L1 distance + sigmoid, threshold 0.4866). Next.js 14 + TypeScript + Tailwind + TensorFlow.js. Model converted via custom convert_tfjs.py (local _tfjs_py/ dir has tensorflowjs 4.14.0; official `pip install tensorflowjs` fails on this machine due to uvloop not supporting Windows).
- [2026-08-03 21:00:50] TensorFlow 2.14.0 was reinstalled successfully (was corrupted). The model.h5 was saved with Keras 3.13.2 which TF 2.14's Keras 2.14 can't deserialize (needs standalone `keras>=3.0` + `safe_mode=False`). However, installing Keras 3 breaks `tensorflow.keras` needed by the local `_tfjs_py/tensorflowjs` 4.14.0. Solution: temporarily install Keras 3, rebuild model architecture, export to SavedModel, uninstall Keras 3, then use `_tfjs_py/tensorflowjs` (with mocked jax/tensorflow_hub/tensorflow_decision_forests) to convert SavedModel → TF.js GraphModel. **Why:** Keras 3 serialization is incompatible with TF 2.14's Keras 2. **How to apply:** The GraphModel output is in public/model/ (model.json + 40 shards, ~160MB). modelLoader.ts now uses tf.loadGraphModel() not tf.loadLayersModel(). No custom layer registration needed.


### Reference

