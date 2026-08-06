/**
 * Node smoke test: load converted model and run inference on synthetic inputs.
 * Usage: node scripts/test-model-load.mjs
 */
import * as tf from '@tensorflow/tfjs';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const modelDir = join(__dirname, '..', 'public', 'model');

class L2NormalizeLayer extends tf.layers.Layer {
  static className = 'L2NormalizeLayer';
  computeOutputShape(inputShape) {
    return inputShape;
  }
  call(inputs) {
    const input = Array.isArray(inputs) ? inputs[0] : inputs;
    return tf.linalg.l2Normalize(input, -1);
  }
}

class L1DistanceLayer extends tf.layers.Layer {
  static className = 'L1DistanceLayer';
  computeOutputShape(inputShape) {
    return Array.isArray(inputShape) && Array.isArray(inputShape[0]) ? inputShape[0] : inputShape;
  }
  call(inputs) {
    const [a, b] = inputs;
    return tf.abs(tf.sub(a, b));
  }
}

tf.serialization.registerClass(L2NormalizeLayer);
tf.serialization.registerClass(L1DistanceLayer);

const modelJson = JSON.parse(readFileSync(join(modelDir, 'model.json'), 'utf8'));
const weights = readFileSync(join(modelDir, 'group1-shard1of1.bin'));

const handler = tf.io.fromMemory({
  modelTopology: modelJson.modelTopology,
  weightSpecs: modelJson.weightsManifest[0].weights,
  weightData: weights.buffer,
});

await tf.ready();
console.log('Loading model from memory...');
const model = await tf.loadLayersModel(handler);
console.log('Model loaded.');
console.log('Inputs:', model.inputs.map((t) => t.shape));
console.log('Outputs:', model.outputs.map((t) => t.shape));

const a = tf.randomUniform([1, 112, 896, 1]);
const b = tf.randomUniform([1, 112, 896, 1]);
const out = model.predict([a, b]);
const arr = await (Array.isArray(out) ? out[0] : out).data();
console.log('Inference score:', arr[0]);

a.dispose();
b.dispose();
(Array.isArray(out) ? out[0] : out).dispose();
model.dispose();
console.log('OK');
