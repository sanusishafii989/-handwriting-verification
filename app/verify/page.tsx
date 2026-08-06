'use client';

import * as React from 'react';
import {
  ArrowRightLeft,
  Trash2,
  Download,
  RotateCcw,
  History,
  ShieldCheck,
  ShieldAlert,
  AlertCircle,
  TrendingUp,
  FileText,
} from 'lucide-react';
import Button from '@/components/ui/Button';
import Card from '@/components/ui/Card';
import UploadArea from '@/components/ui/UploadArea';
import ResultDisplay from '@/components/ui/ResultDisplay';
import ThresholdSlider from '@/components/ui/ThresholdSlider';
import ModelStatus, { type ModelStatusInfo } from '@/components/ui/ModelStatus';
import {
  compareImages,
  exportResultAsJSON,
  storeHistoryItem,
  getHistory,
  clearHistory,
} from '@/lib/similarity';
import { fileToImageData } from '@/lib/preprocess';
import { loadModel, getModelInfo, isModelLoaded } from '@/lib/modelLoader';
import {
  ComparisonResult,
  DEFAULT_THRESHOLD,
  HistoryItem,
  MODEL_VERSION,
} from '@/lib/types';

export default function VerifyPage() {
  const [mounted, setMounted] = React.useState(false);

  const [fileA, setFileA] = React.useState<File | null>(null);
  const [fileB, setFileB] = React.useState<File | null>(null);
  const [previewA, setPreviewA] = React.useState<string | null>(null);
  const [previewB, setPreviewB] = React.useState<string | null>(null);
  const [fileNameA, setFileNameA] = React.useState<string | null>(null);
  const [fileNameB, setFileNameB] = React.useState<string | null>(null);
  const [imageDataA, setImageDataA] = React.useState<ImageData | null>(null);
  const [imageDataB, setImageDataB] = React.useState<ImageData | null>(null);
  const [errorA, setErrorA] = React.useState<string | null>(null);
  const [errorB, setErrorB] = React.useState<string | null>(null);

  const [threshold, setThreshold] = React.useState<number>(DEFAULT_THRESHOLD);
  const [result, setResult] = React.useState<ComparisonResult | null>(null);
  const [isComparing, setIsComparing] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  const [history, setHistory] = React.useState<HistoryItem[]>([]);

  const [modelStatus, setModelStatus] = React.useState<ModelStatusInfo>({
    loaded: false,
    loading: true,
    error: null,
    modelVersion: MODEL_VERSION,
    lastInferenceTime: null,
    totalInferences: 0,
  });

  const inferenceCountRef = React.useRef(0);

  React.useEffect(() => {
    setMounted(true);
  }, []);

  React.useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        await loadModel();
        if (cancelled) return;
        const info = getModelInfo();
        setModelStatus((s) => ({
          ...s,
          loaded: info.loaded,
          loading: false,
          error: info.error,
        }));
      } catch (e) {
        if (cancelled) return;
        setModelStatus((s) => ({
          ...s,
          loaded: false,
          loading: false,
          error: e instanceof Error ? e.message : 'Model failed to load',
        }));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  React.useEffect(() => {
    if (!mounted) return;
    try {
      setHistory(getHistory().slice(0, 5));
    } catch {
      /* ignore */
    }
  }, [mounted]);

  React.useEffect(() => {
    return () => {
      if (previewA) URL.revokeObjectURL(previewA);
      if (previewB) URL.revokeObjectURL(previewB);
    };
  }, [previewA, previewB]);

  const handleFileA = React.useCallback(async (f: File) => {
    setErrorA(null);
    setResult(null);
    setError(null);
    try {
      if (previewA) URL.revokeObjectURL(previewA);
      const url = URL.createObjectURL(f);
      const data = await fileToImageData(f);
      setFileA(f);
      setPreviewA(url);
      setFileNameA(f.name);
      setImageDataA(data);
    } catch (e) {
      setErrorA(e instanceof Error ? e.message : 'Could not decode image');
    }
  }, [previewA]);

  const handleFileB = React.useCallback(async (f: File) => {
    setErrorB(null);
    setResult(null);
    setError(null);
    try {
      if (previewB) URL.revokeObjectURL(previewB);
      const url = URL.createObjectURL(f);
      const data = await fileToImageData(f);
      setFileB(f);
      setPreviewB(url);
      setFileNameB(f.name);
      setImageDataB(data);
    } catch (e) {
      setErrorB(e instanceof Error ? e.message : 'Could not decode image');
    }
  }, [previewB]);

  const handleClearA = React.useCallback(() => {
    if (previewA) URL.revokeObjectURL(previewA);
    setFileA(null);
    setPreviewA(null);
    setFileNameA(null);
    setImageDataA(null);
    setErrorA(null);
    setResult(null);
    setError(null);
  }, [previewA]);

  const handleClearB = React.useCallback(() => {
    if (previewB) URL.revokeObjectURL(previewB);
    setFileB(null);
    setPreviewB(null);
    setFileNameB(null);
    setImageDataB(null);
    setErrorB(null);
    setResult(null);
    setError(null);
  }, [previewB]);

  const handleClearAll = React.useCallback(() => {
    handleClearA();
    handleClearB();
  }, [handleClearA, handleClearB]);

  const handleSwap = React.useCallback(() => {
    setResult(null);
    setError(null);
    setFileA((prevA) => {
      setFileB(prevA);
      return fileB;
    });
    setPreviewA((prevA) => {
      setPreviewB(prevA);
      return previewB;
    });
    setFileNameA((prevA) => {
      setFileNameB(prevA);
      return fileNameB;
    });
    setImageDataA((prevA) => {
      setImageDataB(prevA);
      return imageDataB;
    });
  }, [fileB, previewB, fileNameB, imageDataB]);

  const handleCompare = React.useCallback(async () => {
    if (!fileA || !fileB) return;
    setIsComparing(true);
    setError(null);
    try {
      if (!isModelLoaded()) {
        const info = getModelInfo();
        if (info.error) throw new Error(info.error);
        throw new Error('Model is still loading. Please wait a moment and try again.');
      }
      const res = await compareImages(fileA, fileB, threshold);
      inferenceCountRef.current += 1;
      const stored = storeHistoryItem({
        score: res.score,
        threshold: res.threshold,
        verdict: res.verdict,
        confidence: res.confidence,
        inferenceTimeMs: res.inferenceTimeMs,
        timestamp: res.timestamp,
        fileNameA: fileNameA || 'sample-a.png',
        fileNameB: fileNameB || 'sample-b.png',
        modelVersion: res.modelVersion,
      });
      setResult(res);
      setModelStatus((s) => ({
        ...s,
        loaded: true,
        error: null,
        lastInferenceTime: res.inferenceTimeMs,
        totalInferences: inferenceCountRef.current,
      }));
      setHistory((prev) => {
        const next = [stored, ...prev.filter((x) => x.id !== stored.id)];
        return next.slice(0, 5);
      });
    } catch (e) {
      const msg =
        e instanceof Error ? e.message : 'Unknown error occurred during inference';
      setError(msg);
      const info = getModelInfo();
      if (info.error) {
        setModelStatus((s) => ({ ...s, loaded: false, error: info.error }));
      }
    } finally {
      setIsComparing(false);
    }
  }, [fileA, fileB, threshold, fileNameA, fileNameB]);

  const handleExport = React.useCallback(() => {
    if (!result) return;
    exportResultAsJSON(result, fileNameA || 'sample-a.png', fileNameB || 'sample-b.png');
  }, [result, fileNameA, fileNameB]);

  const handleClearHistory = React.useCallback(() => {
    clearHistory();
    setHistory([]);
  }, []);

  const canCompare = Boolean(fileA && fileB) && !isComparing;
  const currentScore = result?.score ?? null;

  if (!mounted) {
    return (
      <div className="container mx-auto px-4 py-16">
        <div className="max-w-6xl mx-auto h-96 rounded-2xl glass animate-pulse" />
      </div>
    );
  }

  return (
    <div className="relative flex-1 py-8 sm:py-12 lg:py-16">
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-20 -left-20 w-[500px] h-[500px] rounded-full bg-cyan/10 blur-[120px] opacity-50" />
        <div className="absolute top-1/3 -right-20 w-[500px] h-[500px] rounded-full bg-gold/10 blur-[120px] opacity-40" />
      </div>

      <div className="container mx-auto px-4 sm:px-6 lg:px-8 relative">
        <header className="max-w-4xl mx-auto mb-8 sm:mb-12 text-center">
          <p className="text-[11px] uppercase tracking-[0.2em] text-cyan font-semibold mb-3">
            Verification Workbench
          </p>
          <h1 className="text-2xl sm:text-4xl lg:text-5xl font-black tracking-tight text-navy dark:text-white leading-tight mb-4">
            Handwriting Writer Matching
          </h1>
          <p className="text-sm sm:text-base text-navy/60 dark:text-white/60 max-w-2xl mx-auto leading-relaxed">
            Upload two handwritten samples. The Siamese network will extract 64D
            embeddings, compute L1 Manhattan distance, and return a similarity score
            against the tuned decision threshold.
          </p>
        </header>

        <div className="grid grid-cols-1 xl:grid-cols-5 gap-6 lg:gap-8 max-w-7xl mx-auto">
          <div className="xl:col-span-2 space-y-5">
            <Card className="p-5 sm:p-6">
              <div className="flex items-center justify-between mb-5">
                <div>
                  <h3 className="text-base font-bold text-navy dark:text-white">Samples</h3>
                  <p className="text-[11px] text-navy/50 dark:text-white/50">
                    PNG, JPG, JPEG up to 10MB. Centered, 112×896 grayscale internally.
                  </p>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleSwap}
                  disabled={!fileA || !fileB}
                  aria-label="Swap samples"
                  title="Swap samples"
                >
                  <ArrowRightLeft className="w-4 h-4" />
                  Swap
                </Button>
              </div>
              <div className="space-y-4">
                <UploadArea
                  label="Sample A — Query Handwriting"
                  index={1}
                  imageData={imageDataA}
                  previewUrl={previewA}
                  fileName={fileNameA}
                  onFile={handleFileA}
                  onClear={handleClearA}
                  error={errorA}
                />
                <UploadArea
                  label="Sample B — Reference Handwriting"
                  index={2}
                  imageData={imageDataB}
                  previewUrl={previewB}
                  fileName={fileNameB}
                  onFile={handleFileB}
                  onClear={handleClearB}
                  error={errorB}
                />
              </div>
            </Card>

            <Card className="p-5 sm:p-6">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2.5">
                  <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-cyan to-navy-light flex items-center justify-center">
                    <History className="w-4.5 h-4.5 text-white" />
                  </div>
                  <div>
                    <h3 className="text-sm font-bold text-navy dark:text-white">Recent Comparisons</h3>
                    <p className="text-[11px] text-navy/50 dark:text-white/50">
                      Stored locally in your browser
                    </p>
                  </div>
                </div>
                {history.length > 0 && (
                  <button
                    onClick={handleClearHistory}
                    className="text-[11px] text-navy/50 dark:text-white/50 hover:text-red-400 transition-colors flex items-center gap-1"
                    title="Clear history"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                    Clear
                  </button>
                )}
              </div>

              {history.length === 0 ? (
                <div className="rounded-xl bg-navy/[0.02] dark:bg-white/[0.02] border border-navy/5 dark:border-white/5 p-6 text-center">
                  <FileText className="w-8 h-8 text-navy/20 dark:text-white/20 mx-auto mb-3" />
                  <p className="text-xs font-medium text-navy/50 dark:text-white/50 mb-0.5">No history yet</p>
                  <p className="text-[11px] text-navy/35 dark:text-white/35 max-w-xs mx-auto">
                    After you run a comparison, your last 20 results will appear here for quick reference.
                  </p>
                </div>
              ) : (
                <ul className="space-y-2.5 max-h-[380px] overflow-y-auto scrollbar-thin pr-1">
                  {history.map((item) => (
                    <HistoryRow
                      key={item.id}
                      item={item}
                      currentThreshold={threshold}
                    />
                  ))}
                </ul>
              )}
            </Card>
          </div>

          <div className="xl:col-span-3 space-y-5">
            <Card className="p-5 sm:p-6">
              <div className="flex flex-wrap items-center gap-3">
                <Button
                  variant="primary"
                  size="lg"
                  onClick={handleCompare}
                  disabled={!canCompare}
                  isLoading={isComparing}
                  leftIcon={<ShieldCheck className="w-4.5 h-4.5" />}
                >
                  {isComparing ? 'Comparing…' : 'Compare Handwriting'}
                </Button>
                <Button
                  variant="outline"
                  size="lg"
                  onClick={handleExport}
                  disabled={!result || isComparing}
                  leftIcon={<Download className="w-4.5 h-4.5" />}
                >
                  Export JSON
                </Button>
                <Button
                  variant="ghost"
                  size="lg"
                  onClick={handleClearAll}
                  disabled={(!fileA && !fileB && !result) || isComparing}
                  leftIcon={<RotateCcw className="w-4.5 h-4.5" />}
                >
                  Clear
                </Button>
                <div className="ml-auto hidden sm:flex items-center gap-2 text-[11px] text-navy/45 dark:text-white/45 font-mono">
                  <TrendingUp className="w-3.5 h-3.5 text-cyan" />
                  <span>Threshold t = {threshold.toFixed(4)}</span>
                </div>
              </div>
            </Card>

            <ThresholdSlider
              threshold={threshold}
              onChange={(v) => {
                setThreshold(v);
                setResult((prev) => {
                  if (!prev) return prev;
                  const nextVerdict = prev.score >= v ? 'Same Writer' : 'Different Writer';
                  const maxDistance = Math.max(v, 1 - v);
                  const confidence = Math.min(
                    1,
                    Math.max(0, Math.abs(prev.score - v) / Math.max(maxDistance, 1e-6)),
                  );
                  return {
                    ...prev,
                    threshold: v,
                    verdict: nextVerdict,
                    confidence,
                  };
                });
              }}
              currentScore={currentScore}
            />

            {error && (
              <div className="rounded-2xl p-5 border border-red-500/30 bg-red-500/[0.05] flex items-start gap-3">
                <AlertCircle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm font-semibold text-red-300">Inference failed</p>
                  <p className="text-xs text-red-300/70 mt-1 leading-relaxed">{error}</p>
                </div>
              </div>
            )}

            <ResultDisplay result={result} isLoading={isComparing} error={null} />

            <ModelStatus status={modelStatus} compact />
          </div>
        </div>
      </div>
    </div>
  );
}

function HistoryRow({
  item,
  currentThreshold,
}: {
  item: HistoryItem;
  currentThreshold: number;
}) {
  const isSame = item.verdict === 'Same Writer';
  const scorePct = (item.score * 100).toFixed(1);
  const threshPct = (item.threshold * 100).toFixed(1);
  const threshChanged = Math.abs(item.threshold - currentThreshold) > 1e-6;
  const liveVerdict =
    item.score >= currentThreshold ? 'Same Writer' : 'Different Writer';
  const liveFlipped = liveVerdict !== item.verdict;

  return (
    <li className="group rounded-xl bg-navy/[0.02] dark:bg-white/[0.02] border border-navy/5 dark:border-white/5 hover:border-cyan/20 hover:bg-navy/[0.03] dark:hover:bg-white/[0.03] transition-all p-3.5">
      <div className="flex items-start justify-between gap-3 mb-2.5">
        <div className="flex items-center gap-2.5 min-w-0">
          <div
            className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 ${
              isSame
                ? 'bg-green-500/15 border border-green-500/30'
                : 'bg-orange-500/15 border border-orange-500/30'
            }`}
          >
            {isSame ? (
              <ShieldCheck className="w-4 h-4 text-green-400" />
            ) : (
              <ShieldAlert className="w-4 h-4 text-orange-400" />
            )}
          </div>
          <div className="min-w-0">
            <p
              className={`text-sm font-bold leading-tight ${
                isSame ? 'text-green-400' : 'text-orange-400'
              }`}
            >
              {item.verdict}
            </p>
            <p className="text-[10px] text-navy/45 dark:text-white/45 truncate max-w-[240px]">
              {item.fileNameA} vs {item.fileNameB}
            </p>
          </div>
        </div>
        <span className="text-[10px] text-navy/40 dark:text-white/40 font-mono shrink-0">
          {formatRelative(item.timestamp)}
        </span>
      </div>

      <div className="relative h-2 rounded-full bg-navy/5 dark:bg-white/5 overflow-hidden mb-2">
        <div
          className={`absolute inset-y-0 left-0 rounded-full ${
            isSame
              ? 'bg-gradient-to-r from-cyan/70 to-green-400/80'
              : 'bg-gradient-to-r from-red-500/70 to-orange-400/80'
          }`}
          style={{ width: `${scorePct}%` }}
        />
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-gold z-10"
          style={{ left: `${threshPct}%` }}
        />
      </div>

      <div className="flex items-center justify-between text-[10px] font-mono text-navy/50 dark:text-white/50">
        <span>
          score <span className="text-navy/80 dark:text-white/80">{item.score.toFixed(4)}</span>
        </span>
        <span>
          t = <span className={threshChanged ? 'text-gold' : 'text-navy/80 dark:text-white/80'}>
            {item.threshold.toFixed(4)}
          </span>
          {threshChanged && (
            <span className="text-navy/30 dark:text-white/30 ml-1.5">
              now {currentThreshold.toFixed(4)}
              {liveFlipped && (
                <span className={`ml-1.5 ${liveVerdict === 'Same Writer' ? 'text-green-400' : 'text-orange-400'}`}>
                  → {liveVerdict}
                </span>
              )}
            </span>
          )}
        </span>
        <span>{item.inferenceTimeMs.toFixed(0)} ms</span>
      </div>
    </li>
  );
}

function formatRelative(iso: string): string {
  try {
    const then = new Date(iso).getTime();
    const diff = Date.now() - then;
    const s = Math.max(1, Math.floor(diff / 1000));
    if (s < 60) return `${s}s ago`;
    const m = Math.floor(s / 60);
    if (m < 60) return `${m}m ago`;
    const h = Math.floor(m / 60);
    if (h < 24) return `${h}h ago`;
    const d = Math.floor(h / 24);
    return `${d}d ago`;
  } catch {
    return iso;
  }
}
