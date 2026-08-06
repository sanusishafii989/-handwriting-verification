'use client';

import { Gauge, RotateCcw, Info } from 'lucide-react';

interface ThresholdSliderProps {
  threshold: number;
  onChange: (value: number) => void;
  currentScore?: number | null;
}

const DEFAULT_THRESHOLD = 0.4866;

export default function ThresholdSlider({
  threshold,
  onChange,
  currentScore = null,
}: ThresholdSliderProps) {
  const verdict =
    currentScore !== null
      ? currentScore >= threshold
        ? 'Same Writer'
        : 'Different Writer'
      : null;

  const isVerdictChange =
    currentScore !== null &&
    ((currentScore >= DEFAULT_THRESHOLD && currentScore < threshold) ||
      (currentScore < DEFAULT_THRESHOLD && currentScore >= threshold));

  return (
    <div className="rounded-2xl glass p-5 sm:p-6 w-full">
      <div className="flex items-start justify-between mb-5">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gold/15 flex items-center justify-center">
            <Gauge className="w-4.5 h-4.5 text-gold" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-navy dark:text-white">Decision Threshold</h3>
            <p className="text-[11px] text-navy/50 dark:text-white/50">
              Tuned on validation set. Default: <span className="font-mono text-gold">{DEFAULT_THRESHOLD.toFixed(4)}</span>
            </p>
          </div>
        </div>
        <button
          onClick={() => onChange(DEFAULT_THRESHOLD)}
          disabled={Math.abs(threshold - DEFAULT_THRESHOLD) < 0.0001}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[11px] font-medium text-cyan hover:bg-cyan/10 transition-all disabled:opacity-30 disabled:cursor-not-allowed"
          title="Reset to default"
        >
          <RotateCcw className="w-3 h-3" />
          Reset
        </button>
      </div>

      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <span className="text-xs text-navy/60 dark:text-white/60 font-medium">Sensitivity</span>
          <div className="flex items-center gap-2">
            <span className="text-2xl font-mono font-black text-gold tracking-tight">
              {threshold.toFixed(4)}
            </span>
          </div>
          <span className="text-xs text-navy/60 dark:text-white/60 font-medium">Specificity</span>
        </div>

        <div className="relative pt-2">
          <input
            type="range"
            min={0}
            max={1}
            step={0.0001}
            value={threshold}
            onChange={(e) => onChange(parseFloat(e.target.value))}
            className="relative z-10"
          />
          <div className="flex justify-between mt-2 text-[10px] font-mono text-navy/30 dark:text-white/30">
            <span>0.0000</span>
            <span>0.2500</span>
            <span className="text-gold/70">0.4866</span>
            <span>0.7500</span>
            <span>1.0000</span>
          </div>
          <div
            className="absolute top-2 h-1.5 rounded-full bg-gold/30 pointer-events-none"
            style={{
              left: '0%',
              width: `${threshold * 100}%`,
              marginTop: '4px',
            }}
          />
        </div>

        {currentScore !== null && (
          <div
            className={`rounded-xl p-4 border transition-all duration-300 ${
              isVerdictChange
                ? 'bg-gold/10 border-gold/40'
                : 'bg-navy/[0.02] dark:bg-white/[0.02] border-navy/5 dark:border-white/5'
            }`}
          >
            <div className="flex items-start justify-between gap-4">
              <div className="flex items-center gap-2.5">
                {isVerdictChange ? (
                  <Info className="w-4 h-4 text-gold shrink-0 mt-0.5 animate-pulse" />
                ) : (
                  <Info className="w-4 h-4 text-navy/30 dark:text-white/30 shrink-0 mt-0.5" />
                )}
                <div>
                  <p className="text-xs font-medium text-navy/80 dark:text-white/80">
                    Live Verdict Preview
                  </p>
                  <p className="text-[11px] text-navy/50 dark:text-white/50 mt-0.5">
                    Score {currentScore.toFixed(4)} vs threshold {threshold.toFixed(4)}
                  </p>
                </div>
              </div>
              <div
                className={`px-3 py-1.5 rounded-lg text-xs font-bold whitespace-nowrap ${
                  verdict === 'Same Writer'
                    ? 'bg-green-500/15 text-green-600 dark:text-green-400 border border-green-500/30'
                    : 'bg-orange-500/15 text-orange-600 dark:text-orange-400 border border-orange-500/30'
                }`}
              >
                {verdict}
              </div>
            </div>
            {isVerdictChange && (
              <p className="mt-3 text-[11px] text-gold/80 pl-6.5">
                Changing the threshold from the default value has flipped the verdict. Use with caution.
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
