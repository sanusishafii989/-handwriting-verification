import * as tf from '@tensorflow/tfjs';
import { loadModel } from './modelLoader';
import { fileToImageData, preprocessImage } from './preprocess';
import {
  ComparisonResult,
  ExportPayload,
  HistoryItem,
  Verdict,
  DEFAULT_THRESHOLD,
  MODEL_VERSION,
  MODEL_ARCHITECTURE,
} from './types';

const HISTORY_KEY = 'handwriting_verification_history';
const HISTORY_MAX = 20;
const EXPORT_VERSION = '1.0';

export async function compareImages(
  fileA: File,
  fileB: File,
  threshold: number = DEFAULT_THRESHOLD,
): Promise<ComparisonResult> {
  const startTs = performance.now();

  const [imgDataA, imgDataB, model] = await Promise.all([
    fileToImageData(fileA),
    fileToImageData(fileB),
    loadModel(),
  ]);

  let tensorA: tf.Tensor4D | null = null;
  let tensorB: tf.Tensor4D | null = null;
  let prediction: tf.Tensor | null = null;
  let score = 0;

  try {
    tensorA = preprocessImage(imgDataA);
    tensorB = preprocessImage(imgDataB);

    const predictionOut = tf.tidy(() => {
      // GraphModel.execute returns NamedTensor(s)
      const result = model.execute([tensorA!, tensorB!]);
      return result;
    });

    prediction = Array.isArray(predictionOut) ? predictionOut[0] : predictionOut;
    const data = await prediction.data();
    score = Number(data[0]);
    if (!Number.isFinite(score)) score = 0;
    score = Math.min(1, Math.max(0, score));
  } finally {
    if (tensorA) tensorA.dispose();
    if (tensorB) tensorB.dispose();
    if (prediction) prediction.dispose();
  }

  const inferenceTimeMs = performance.now() - startTs;
  const verdict: Verdict = score >= threshold ? 'Same Writer' : 'Different Writer';

  const maxDistance = Math.max(threshold, 1 - threshold);
  const rawConfidence = Math.abs(score - threshold) / Math.max(maxDistance, 1e-6);
  const confidence = Math.min(1, Math.max(0, rawConfidence));

  const result: ComparisonResult = {
    score,
    threshold,
    verdict,
    confidence,
    inferenceTimeMs,
    modelVersion: MODEL_VERSION,
    timestamp: new Date().toISOString(),
  };

  return result;
}

export function buildExportPayload(
  result: ComparisonResult,
  fileNameA: string,
  fileNameB: string,
): ExportPayload {
  return {
    exportVersion: EXPORT_VERSION,
    modelVersion: result.modelVersion,
    modelArchitecture: {
      backbone: MODEL_ARCHITECTURE.backbone,
      embedding: MODEL_ARCHITECTURE.embedding,
      distance: MODEL_ARCHITECTURE.distance,
      inputShape: [112, 896, 1],
    },
    score: result.score,
    threshold: result.threshold,
    verdict: result.verdict,
    confidence: result.confidence,
    inferenceTimeMs: result.inferenceTimeMs,
    timestamp: result.timestamp,
    files: {
      sampleA: fileNameA,
      sampleB: fileNameB,
    },
  };
}

export function exportResultAsJSON(
  result: ComparisonResult,
  fileNameA: string,
  fileNameB: string,
): void {
  if (typeof document === 'undefined') return;

  const payload = buildExportPayload(result, fileNameA, fileNameB);
  const blob = new Blob([JSON.stringify(payload, null, 2)], {
    type: 'application/json',
  });
  const url = URL.createObjectURL(blob);
  const stamp = result.timestamp.replace(/[:.]/g, '-');
  const a = document.createElement('a');
  a.href = url;
  a.download = `handwriting-result-${stamp}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

function safeReadHistory(): HistoryItem[] {
  if (typeof localStorage === 'undefined') return [];
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((x) => x && typeof x.id === 'string') as HistoryItem[];
  } catch {
    return [];
  }
}

function safeWriteHistory(items: HistoryItem[]): void {
  if (typeof localStorage === 'undefined') return;
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(items.slice(0, HISTORY_MAX)));
  } catch {
    /* ignore quota errors */
  }
}

export function getHistory(): HistoryItem[] {
  return safeReadHistory();
}

export function storeHistoryItem(extra: Omit<HistoryItem, 'id'>): HistoryItem {
  const item: HistoryItem = {
    id:
      (typeof crypto !== 'undefined' && 'randomUUID' in crypto
        ? crypto.randomUUID()
        : `h-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`),
    ...extra,
  };
  const current = safeReadHistory();
  const next = [item, ...current].slice(0, HISTORY_MAX);
  safeWriteHistory(next);
  return item;
}

export function clearHistory(): void {
  safeWriteHistory([]);
}

export function removeHistoryItem(id: string): HistoryItem[] {
  const current = safeReadHistory();
  const next = current.filter((x) => x.id !== id);
  safeWriteHistory(next);
  return next;
}
