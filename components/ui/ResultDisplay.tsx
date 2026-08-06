'use client';

import { CheckCircle2, XCircle, Target, Clock, ShieldCheck, ShieldAlert, Gauge, TrendingUp } from 'lucide-react';
import Card from './Card';

export interface ComparisonResult {
  score: number;
  threshold: number;
  verdict: 'Same Writer' | 'Different Writer';
  confidence: number;
  inferenceTimeMs: number;
  modelVersion: string;
  timestamp: string;
}

interface ResultDisplayProps {
  result: ComparisonResult | null;
  isLoading: boolean;
  error?: string | null;
}

export default function ResultDisplay({
  result,
  isLoading,
  error = null,
}: ResultDisplayProps) {
  if (isLoading) {
    return (
      <Card className="w-full p-8">
        <div className="flex flex-col items-center justify-center py-8">
          <div className="relative w-20 h-20 mb-6">
            <div className="absolute inset-0 rounded-full border-4 border-cyan/20" />
            <div className="absolute inset-0 rounded-full border-4 border-t-cyan border-r-transparent border-b-transparent border-l-transparent animate-spin" />
            <div className="absolute inset-3 rounded-full border-4 border-b-gold border-r-gold/30 border-t-transparent border-l-transparent animate-spin" style={{ animationDirection: 'reverse', animationDuration: '1.5s' }} />
          </div>
          <h4 className="text-navy dark:text-white font-semibold mb-1.5">Running Inference</h4>
          <p className="text-sm text-navy/50 dark:text-white/50 text-center max-w-xs">
            Siamese network is comparing handwriting features using Xception backbone...
          </p>
          <div className="mt-6 flex items-center gap-2 text-xs text-cyan/70">
            <Gauge className="w-3.5 h-3.5 animate-pulse" />
            <span>Extracting 64D embeddings & calculating L1 distance</span>
          </div>
        </div>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="w-full p-8 border-red-500/30 bg-red-500/5">
        <div className="flex flex-col items-center justify-center py-4 text-center">
          <div className="w-16 h-16 rounded-2xl bg-red-500/10 flex items-center justify-center mb-4">
            <XCircle className="w-8 h-8 text-red-400" />
          </div>
          <h4 className="text-navy dark:text-white font-semibold mb-1">Inference Failed</h4>
          <p className="text-sm text-navy/60 dark:text-white/60 max-w-sm">{error}</p>
        </div>
      </Card>
    );
  }

  if (!result) {
    return (
      <Card className="w-full p-8">
        <div className="flex flex-col items-center justify-center py-8 text-center">
          <div className="w-20 h-20 rounded-2xl bg-navy/[0.02] dark:bg-white/[0.02] border border-navy/5 dark:border-white/5 flex items-center justify-center mb-5">
            <Target className="w-10 h-10 text-navy/20 dark:text-white/20" />
          </div>
          <h4 className="text-navy/70 dark:text-white/70 font-medium mb-1.5">Awaiting Comparison</h4>
          <p className="text-sm text-navy/40 dark:text-white/40 max-w-xs leading-relaxed">
            Upload two handwriting samples and click <span className="text-cyan font-medium">Compare</span> to
            verify if they were written by the same person.
          </p>
        </div>
      </Card>
    );
  }

  const isSame = result.verdict === 'Same Writer';
  const scorePercent = (result.score * 100).toFixed(2);
  const thresholdPercent = (result.threshold * 100).toFixed(2);

  return (
    <Card className="w-full p-6 sm:p-8 animate-slide-up">
      <div className="flex items-center justify-between mb-7">
        <div className="flex items-center gap-3">
          <div
            className={`w-11 h-11 rounded-2xl flex items-center justify-center shadow-lg ${
              isSame
                ? 'bg-gradient-to-br from-green-500 to-emerald-600 shadow-green-500/30'
                : 'bg-gradient-to-br from-orange-500 to-red-500 shadow-red-500/30'
            }`}
          >
            {isSame ? (
              <ShieldCheck className="w-5.5 h-5.5 text-white" />
            ) : (
              <ShieldAlert className="w-5.5 h-5.5 text-white" />
            )}
          </div>
          <div>
            <p className="text-[11px] uppercase tracking-wider text-navy/40 dark:text-white/40 font-semibold mb-0.5">
              Verdict
            </p>
            <h3 className={`text-xl font-bold ${isSame ? 'text-green-600 dark:text-green-400' : 'text-orange-600 dark:text-orange-400'}`}>
              {result.verdict}
            </h3>
          </div>
        </div>
        <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-navy/5 dark:bg-white/5 text-[11px] text-navy/60 dark:text-white/60 font-mono">
          <Clock className="w-3 h-3" />
          {result.inferenceTimeMs.toFixed(0)} ms
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-7">
        <div className="sm:col-span-2 relative rounded-2xl bg-gradient-to-br from-navy/[0.02] dark:from-white/[0.02] to-navy/[0.005] dark:to-white/[0.005] border border-navy/5 dark:border-white/5 p-5 overflow-hidden">
          <div className="absolute top-4 right-4 flex items-center gap-1.5 text-[10px] font-mono text-navy/40 dark:text-white/40">
            <TrendingUp className="w-3 h-3" />
            score
          </div>
          <p className="text-[11px] uppercase tracking-wider text-cyan/70 font-semibold mb-1">
            Similarity Score
          </p>
          <div className="flex items-baseline gap-1 mb-4">
            <span
              className={`text-5xl font-black tracking-tight ${
                isSame ? 'text-green-600 dark:text-green-400' : 'text-orange-600 dark:text-orange-400'
              }`}
            >
              {scorePercent}
            </span>
            <span className="text-xl font-bold text-navy/30 dark:text-white/30">%</span>
          </div>

          <div className="space-y-3">
            <div className="relative h-4 rounded-full bg-navy/5 dark:bg-white/5 overflow-hidden">
              <div
                className={`absolute inset-y-0 left-0 rounded-full transition-all duration-1000 ease-out ${
                  isSame
                    ? 'bg-gradient-to-r from-cyan via-green-400 to-emerald-500'
                    : 'bg-gradient-to-r from-red-500 via-orange-500 to-amber-500'
                }`}
                style={{ width: `${Math.max(parseFloat(scorePercent), 1)}%` }}
              />
              <div
                className="absolute top-0 bottom-0 w-0.5 bg-gold shadow-[0_0_8px_rgba(255,183,3,0.8)] z-10"
                style={{ left: `${thresholdPercent}%` }}
                title={`Threshold: ${result.threshold.toFixed(4)}`}
              />
              <div
                className="absolute -bottom-5 w-16 -translate-x-1/2 text-[9px] font-mono text-gold whitespace-nowrap text-center"
                style={{ left: `${thresholdPercent}%` }}
              >
                t = {result.threshold.toFixed(4)}
              </div>
            </div>
            <div className="flex justify-between text-[10px] font-mono text-navy/30 dark:text-white/30 pt-4">
              <span>0.00 — Different</span>
              <span className="text-gold font-semibold">Threshold {thresholdPercent}%</span>
              <span>1.00 — Same</span>
            </div>
          </div>
        </div>

        <div className="space-y-3">
          <div className="rounded-2xl bg-navy/[0.02] dark:bg-white/[0.02] border border-navy/5 dark:border-white/5 p-4">
            <p className="text-[10px] uppercase tracking-wider text-navy/40 dark:text-white/40 font-semibold mb-2">
              Confidence Margin
            </p>
            <div className="flex items-baseline gap-1">
              <span className="text-3xl font-black text-navy dark:text-white">
                {(result.confidence * 100).toFixed(1)}
              </span>
              <span className="text-sm font-bold text-navy/30 dark:text-white/30">%</span>
            </div>
            <div className="mt-2 h-1.5 rounded-full bg-navy/5 dark:bg-white/5 overflow-hidden">
              <div
                className="h-full rounded-full bg-gradient-to-r from-cyan to-gold"
                style={{ width: `${result.confidence * 100}%` }}
              />
            </div>
          </div>

          <div className="rounded-2xl bg-navy/[0.02] dark:bg-white/[0.02] border border-navy/5 dark:border-white/5 p-4">
            <p className="text-[10px] uppercase tracking-wider text-navy/40 dark:text-white/40 font-semibold mb-2">
              Threshold (tuned)
            </p>
            <p className="text-2xl font-mono font-bold text-gold">
              {result.threshold.toFixed(4)}
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 p-4 rounded-2xl bg-gradient-to-br from-navy/10 dark:from-navy/20 to-slate-100 dark:to-dark-surface/30 border border-navy/5 dark:border-white/5">
        <StatItem label="Model" value="Siamese-Xcep" sub={result.modelVersion} />
        <StatItem label="Backbone" value="Xception" sub="ImageNet Pretrained" />
        <StatItem label="Embedding" value="64-Dim" sub="L2 Normalized" />
        <StatItem label="Distance" value="L1 + Sigmoid" sub="Manhattan" />
      </div>
    </Card>
  );
}

function StatItem({ label, value, sub }: { label: string; value: string; sub: string }) {
  return (
    <div>
      <p className="text-[9px] uppercase tracking-wider text-navy/35 dark:text-white/35 font-semibold mb-1">{label}</p>
      <p className="text-sm font-bold text-navy dark:text-white">{value}</p>
      <p className="text-[10px] text-navy/40 dark:text-white/40 mt-0.5">{sub}</p>
    </div>
  );
}
