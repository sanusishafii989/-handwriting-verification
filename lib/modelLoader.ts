import * as tf from '@tensorflow/tfjs';
import { MODEL_VERSION } from './types';

let model: tf.GraphModel | null = null;
let loadPromise: Promise<tf.GraphModel> | null = null;
let loadError: string | null = null;

const MODEL_PATH = '/model/model.json';

export async function loadModel(forceReload = false): Promise<tf.GraphModel> {
  if (forceReload) {
    model = null;
    loadPromise = null;
    loadError = null;
  }

  if (model) {
    return model;
  }

  if (loadPromise) {
    return loadPromise;
  }

  loadPromise = (async () => {
    try {
      await tf.ready();

      const loaded = await tf.loadGraphModel(MODEL_PATH, {
        requestInit: { cache: 'no-cache' },
      });

      model = loaded;
      loadError = null;
      return model;
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : 'Unknown error loading model. Ensure model.json and .bin files exist in /public/model/';

      loadError = message;
      model = null;
      throw new Error(message);
    } finally {
      loadPromise = null;
    }
  })();

  return loadPromise;
}

export function getModel(): tf.GraphModel | null {
  return model;
}

export function isModelLoaded(): boolean {
  return model !== null;
}

export function getModelLoadError(): string | null {
  return loadError;
}

export function getModelInfo() {
  return {
    loaded: isModelLoaded(),
    error: getModelLoadError(),
    version: MODEL_VERSION,
    summary: {
      backbone: 'Xception',
      embedding: '64D L2-Normalized',
      distance: 'L1 (Manhattan) + Sigmoid',
      inputShape: [112, 896, 1],
      pretrained: 'ImageNet',
    },
  };
}

export function disposeModel() {
  if (model) {
    model.dispose();
    model = null;
  }
  loadPromise = null;
  loadError = null;
}
