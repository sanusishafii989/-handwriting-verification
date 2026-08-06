'use client';

import { useEffect, useState } from 'react';
import { Cpu, CheckCircle2, Loader2, AlertTriangle, XCircle, Clock, Database, Layers } from 'lucide-react';
import Card from './Card';

export interface ModelStatusInfo {
  loaded: boolean;
  loading: boolean;
  error: string | null;
  modelVersion: string;
  lastInferenceTime: number | null;
  totalInferences: number;
}

interface ModelStatusProps {
  status: ModelStatusInfo;
  compact?: boolean;
}

export default function ModelStatus({ status, compact = false }: ModelStatusProps) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) return null;

  const StatusBadge = () => {
    if (status.loading) {
      return (
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-cyan/10 border border-cyan/30">
          <Loader2 className="w-3.5 h-3.5 text-cyan animate-spin" />
          <span className="text-[11px] font-semibold text-cyan">Loading Model</span>
        </div>
      );
    }
    if (status.error) {
      return (
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-red-500/10 border border-red-500/30">
          <XCircle className="w-3.5 h-3.5 text-red-400" />
          <span className="text-[11px] font-semibold text-red-400">Load Failed</span>
        </div>
      );
    }
    if (status.loaded) {
      return (
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-green-500/10 border border-green-500/30">
          <div className="relative">
            <CheckCircle2 className="w-3.5 h-3.5 text-green-400" />
            <span className="absolute inset-0 rounded-full bg-green-400/50 animate-ping opacity-75" />
          </div>
          <span className="text-[11px] font-semibold text-green-400">Model Ready</span>
        </div>
      );
    }
    return (
      <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-navy/5 dark:bg-white/5 border border-navy/10 dark:border-white/10">
        <AlertTriangle className="w-3.5 h-3.5 text-navy/50 dark:text-white/50" />
        <span className="text-[11px] font-semibold text-navy/50 dark:text-white/50">Not Loaded</span>
      </div>
    );
  };

  if (compact) {
    return (
      <div className="flex items-center justify-between px-4 py-3 rounded-xl bg-navy/[0.02] dark:bg-white/[0.02] border border-navy/5 dark:border-white/5">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-navy to-cyan/70 flex items-center justify-center">
            <Cpu className="w-4 h-4 text-white" />
          </div>
          <div>
            <p className="text-[11px] text-navy/50 dark:text-white/50 leading-tight">Model</p>
            <p className="text-xs font-bold text-navy dark:text-white font-mono leading-tight">Siamese Xception</p>
          </div>
        </div>
        <StatusBadge />
      </div>
    );
  }

  return (
    <Card className="w-full p-5 sm:p-6">
      <div className="flex items-start justify-between mb-5">
        <div className="flex items-center gap-3">
          <div className="w-11 h-11 rounded-2xl bg-gradient-to-br from-navy via-cyan/70 to-cyan flex items-center justify-center shadow-glow">
            <Cpu className="w-5.5 h-5.5 text-white" />
          </div>
          <div>
            <h3 className="text-base font-bold text-navy dark:text-white">Model Status</h3>
            <p className="text-[11px] text-navy/50 dark:text-white/50">Siamese Network with Xception Backbone</p>
          </div>
        </div>
        <StatusBadge />
      </div>

      {status.error && (
        <div className="mb-5 p-4 rounded-xl bg-red-500/5 border border-red-500/20">
          <p className="text-[11px] uppercase tracking-wider text-red-400/80 font-semibold mb-1">
            Error
          </p>
          <p className="text-xs text-red-300/80">{status.error}</p>
          <p className="text-[10px] text-navy/40 dark:text-white/40 mt-2 leading-relaxed">
            Ensure <code className="bg-navy/5 dark:bg-white/5 px-1.5 py-0.5 rounded text-cyan font-mono">/public/model/model.json</code> and weights exist.
            See README for conversion instructions.
          </p>
        </div>
      )}

      <div className="grid grid-cols-2 gap-3">
        <div className="p-3.5 rounded-xl bg-navy/[0.02] dark:bg-white/[0.02] border border-navy/5 dark:border-white/5">
          <div className="flex items-center gap-2 mb-1.5">
            <Database className="w-3.5 h-3.5 text-cyan/70" />
            <span className="text-[10px] uppercase tracking-wider text-navy/40 dark:text-white/40 font-semibold">
              Version
            </span>
          </div>
          <p className="text-sm font-mono font-bold text-navy dark:text-white">{status.modelVersion}</p>
        </div>

        <div className="p-3.5 rounded-xl bg-navy/[0.02] dark:bg-white/[0.02] border border-navy/5 dark:border-white/5">
          <div className="flex items-center gap-2 mb-1.5">
            <Clock className="w-3.5 h-3.5 text-gold/70" />
            <span className="text-[10px] uppercase tracking-wider text-navy/40 dark:text-white/40 font-semibold">
              Last Inference
            </span>
          </div>
          <p className="text-sm font-mono font-bold text-navy dark:text-white">
            {status.lastInferenceTime !== null
              ? `${status.lastInferenceTime.toFixed(0)} ms`
              : '—'}
          </p>
        </div>

        <div className="col-span-2 p-3.5 rounded-xl bg-gradient-to-br from-navy/10 dark:from-navy/20 to-slate-100 dark:to-dark-surface/40 border border-navy/5 dark:border-white/5">
          <div className="flex items-center gap-2 mb-2.5">
            <Layers className="w-3.5 h-3.5 text-cyan/80" />
            <span className="text-[10px] uppercase tracking-wider text-navy/40 dark:text-white/40 font-semibold">
              Architecture Summary
            </span>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px]">
            <ArchChip label="Backbone" value="Xception" color="cyan" />
            <ArchChip label="Weights" value="ImageNet" color="gold" />
            <ArchChip label="Embedding" value="64D L2" color="cyan" />
            <ArchChip label="Distance" value="L1 + Sigmoid" color="gold" />
          </div>
        </div>
      </div>

      <div className="mt-4 pt-4 border-t border-navy/5 dark:border-white/5 flex items-center justify-between">
        <span className="text-[10px] text-navy/40 dark:text-white/40 font-mono">
          Input: [112, 896, 1] Grayscale
        </span>
        <span className="text-[10px] text-navy/40 dark:text-white/40 font-mono">
          Inferences: {status.totalInferences.toLocaleString()}
        </span>
      </div>
    </Card>
  );
}

function ArchChip({ label, value, color }: { label: string; value: string; color: 'cyan' | 'gold' }) {
  return (
    <div className="rounded-lg bg-navy/[0.02] dark:bg-white/[0.02] border border-navy/5 dark:border-white/5 px-2.5 py-2">
      <p className="text-[9px] uppercase tracking-wider text-navy/35 dark:text-white/35 mb-0.5">{label}</p>
      <p className={`text-[11px] font-bold ${color === 'cyan' ? 'text-cyan-light' : 'text-gold'}`}>
        {value}
      </p>
    </div>
  );
}
