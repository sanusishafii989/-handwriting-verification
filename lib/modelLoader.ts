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

      // Use absolute URL for compatibility with deployed environments
      const baseUrl = typeof window !== 'undefined' ? window.location.origin : '';
      const modelUrl = `${baseUrl}${MODEL_PATH}`;

      console.log('[ModelLoader] Loading GraphModel from:', modelUrl);

      const loaded = await tf.loadGraphModel(modelUrl, {
        requestInit: { cache: 'no-cache' },
      });

      console.log('[ModelLoader] Model loaded successfully');
      console.log('[ModelLoader] Model inputs:', loaded.inputs);
      console.log('[ModelLoader] Model outputs:', loaded.outputs);

      model = loaded;
      loadError = null;
      return model;
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : 'Unknown error loading model. Ensure model.json and .bin files exist in /public/model/';

      console.error('[ModelLoader] Failed to load model:', message);
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
