import * as tf from '@tensorflow/tfjs';
import { MODEL_VERSION } from './types';

let model: tf.GraphModel | null = null;
let loadPromise: Promise<tf.GraphModel> | null = null;
let loadError: string | null = null;

const MODEL_PATH = '/model/model.json';

export async function loadModel(forceReload = false): Promise<tf.GraphModel> {
  if (forceReload) {
    if (model) {
      model.dispose();
    }
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
      // Explicitly set backend before loading
      try {
        await tf.setBackend('webgl');
      } catch {
        await tf.setBackend('cpu');
      }
      await tf.ready();

      // Build absolute URL — works locally and on Vercel
      const baseUrl = typeof window !== 'undefined' ? window.location.origin : '';
      const modelUrl = `${baseUrl}${MODEL_PATH}`;

      console.log('[ModelLoader] Loading GraphModel from:', modelUrl);
      console.log('[ModelLoader] Backend:', tf.getBackend());

      const loaded = await tf.loadGraphModel(modelUrl);

      console.log('[ModelLoader] Model loaded successfully');
      console.log('[ModelLoader] Inputs:', loaded.inputs.map((i: any) => ({ name: i.name, shape: i.shape, dtype: i.dtype })));
      console.log('[ModelLoader] Outputs:', loaded.outputs.map((o: any) => ({ name: o.name, shape: o.shape, dtype: o.dtype })));

      model = loaded;
      loadError = null;
      return model;
    } catch (err) {
      const rawMessage = err instanceof Error ? err.message : String(err);
      console.error('[ModelLoader] Failed to load model:', rawMessage);
      if (err instanceof Error && err.stack) {
        console.error('[ModelLoader] Stack:', err.stack);
      }
      loadError = rawMessage;
      model = null;
      throw new Error(rawMessage);
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
