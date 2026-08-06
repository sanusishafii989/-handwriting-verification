export type Verdict = 'Same Writer' | 'Different Writer';

export interface ComparisonResult {
  score: number;
  threshold: number;
  verdict: Verdict;
  confidence: number;
  inferenceTimeMs: number;
  modelVersion: string;
  timestamp: string;
}

export interface HistoryItem {
  id: string;
  score: number;
  threshold: number;
  verdict: Verdict;
  confidence: number;
  inferenceTimeMs: number;
  timestamp: string;
  fileNameA: string;
  fileNameB: string;
  modelVersion: string;
}

export interface ExportPayload {
  exportVersion: string;
  modelVersion: string;
  modelArchitecture: {
    backbone: string;
    embedding: string;
    distance: string;
    inputShape: [number, number, number];
  };
  score: number;
  threshold: number;
  verdict: Verdict;
  confidence: number;
  inferenceTimeMs: number;
  timestamp: string;
  files: {
    sampleA: string;
    sampleB: string;
  };
}

export const DEFAULT_THRESHOLD = 0.4866;
export const MODEL_VERSION = 'siamese-xception-v1.0-epoch69';
export const MODEL_ARCHITECTURE = {
  backbone: 'Xception (ImageNet Pretrained, Frozen)',
  embedding: '64-dimensional, L2-normalized',
  distance: 'L1 (Manhattan) distance + Sigmoid',
  inputShape: [112, 896, 1] as [number, number, number],
};
