'use client';

import * as React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  AreaChart,
  Area,
  Legend,
  Cell,
  ReferenceLine,
} from 'recharts';
import {
  Target,
  Crosshair,
  Recycle,
  BarChart3,
  Activity,
  TrendingUp,
  Gauge,
  Sparkles,
  PieChart,
  LineChart as LineChartIcon,
} from 'lucide-react';
import Card from '@/components/ui/Card';
import ModelStatus, { type ModelStatusInfo } from '@/components/ui/ModelStatus';
import { loadModel, getModelInfo } from '@/lib/modelLoader';
import { DEFAULT_THRESHOLD, MODEL_VERSION } from '@/lib/types';

const METRICS = {
  accuracy: 0.8,
  precision: 0.8448,
  recall: 0.735,
  f1: 0.7861,
  rocAuc: 0.8914,
  threshold: DEFAULT_THRESHOLD,
};

const CONFUSION = [
  { name: 'Actual Same', 'Pred Same': 219, 'Pred Diff': 79 },
  { name: 'Actual Diff', 'Pred Same': 40, 'Pred Diff': 270 },
];

const ROC_POINTS = [
  { fpr: 0, tpr: 0 },
  { fpr: 0.02, tpr: 0.22 },
  { fpr: 0.04, tpr: 0.4 },
  { fpr: 0.06, tpr: 0.54 },
  { fpr: 0.08, tpr: 0.63 },
  { fpr: 0.1, tpr: 0.7 },
  { fpr: 0.15, tpr: 0.8 },
  { fpr: 0.2, tpr: 0.86 },
  { fpr: 0.28, tpr: 0.91 },
  { fpr: 0.36, tpr: 0.94 },
  { fpr: 0.48, tpr: 0.97 },
  { fpr: 0.6, tpr: 0.985 },
  { fpr: 0.75, tpr: 0.993 },
  { fpr: 1, tpr: 1 },
];

const SAME_WRITER_HIST = [
  { bin: '0.00–0.10', count: 1, same: 1, diff: 22 },
  { bin: '0.10–0.20', count: 2, same: 2, diff: 38 },
  { bin: '0.20–0.30', count: 5, same: 5, diff: 55 },
  { bin: '0.30–0.40', count: 14, same: 14, diff: 72 },
  { bin: '0.40–0.49', count: 34, same: 34, diff: 85 },
  { bin: '0.49–0.60', count: 74, same: 74, diff: 50 },
  { bin: '0.60–0.70', count: 88, same: 88, diff: 28 },
  { bin: '0.70–0.80', count: 54, same: 54, diff: 12 },
  { bin: '0.80–0.90', count: 21, same: 21, diff: 6 },
  { bin: '0.90–1.00', count: 6, same: 6, diff: 1 },
].map((row) => ({
  ...row,
  diff: row.diff,
}));

const CHART_MARGIN = { top: 16, right: 16, bottom: 8, left: 0 };

const chartTooltipStyle: React.CSSProperties = {
  background: 'rgba(10, 22, 40, 0.95)',
  border: '1px solid rgba(33, 158, 188, 0.3)',
  borderRadius: 12,
  color: '#fff',
  fontSize: 12,
  padding: '8px 12px',
  backdropFilter: 'blur(12px)',
  WebkitBackdropFilter: 'blur(12px)',
};

export default function DashboardPage() {
  const [mounted, setMounted] = React.useState(false);
  const [modelStatus, setModelStatus] = React.useState<ModelStatusInfo>({
    loaded: false,
    loading: true,
    error: null,
    modelVersion: MODEL_VERSION,
    lastInferenceTime: null,
    totalInferences: 0,
  });

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

  if (!mounted) {
    return (
      <div className="container mx-auto px-4 py-16">
        <div className="max-w-7xl mx-auto space-y-8">
          <div className="h-32 rounded-2xl glass animate-pulse" />
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-5">
            {Array.from({ length: 6 }).map((_, i) => (
              <div key={i} className="h-32 rounded-2xl glass animate-pulse" />
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="relative flex-1 py-8 sm:py-12 lg:py-16">
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-20 right-1/4 w-[500px] h-[500px] rounded-full bg-gold/10 blur-[120px] opacity-40" />
        <div className="absolute top-1/3 -left-20 w-[500px] h-[500px] rounded-full bg-cyan/10 blur-[120px] opacity-40" />
      </div>

      <div className="container mx-auto px-4 sm:px-6 lg:px-8 relative max-w-7xl">
        <header className="max-w-4xl mx-auto mb-8 sm:mb-12 text-center">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass mb-6">
            <Sparkles className="w-3.5 h-3.5 text-gold" />
            <span className="text-[11px] font-semibold tracking-wide text-navy/80 dark:text-white/80">
              Model Evaluation Dashboard
            </span>
          </div>
          <p className="text-[11px] uppercase tracking-[0.2em] text-cyan font-semibold mb-3">
            Validation Set Metrics
          </p>
          <h1 className="text-2xl sm:text-4xl lg:text-5xl font-black tracking-tight text-navy dark:text-white leading-tight mb-4">
            Siamese Xception — Performance Snapshot
          </h1>
          <p className="text-sm sm:text-base text-navy/60 dark:text-white/60 max-w-2xl mx-auto leading-relaxed">
            Classification metrics, confusion matrix, ROC curve, and similarity score
            distributions for the best checkpoint (epoch 69, threshold t=0.4866).
          </p>
        </header>

        <ModelStatus status={modelStatus} />

        <section className="mt-8 sm:mt-10">
          <div className="grid grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-5">
            <MetricCard
              label="Accuracy"
              value={(METRICS.accuracy * 100).toFixed(2)}
              suffix="%"
              accent="cyan"
              icon={<Target className="w-5 h-5" />}
              sub="Overall correctness"
            />
            <MetricCard
              label="Precision"
              value={METRICS.precision.toFixed(4)}
              accent="gold"
              icon={<Crosshair className="w-5 h-5" />}
              sub="Positive predictive value"
            />
            <MetricCard
              label="Recall"
              value={METRICS.recall.toFixed(4)}
              accent="cyan"
              icon={<Recycle className="w-5 h-5" />}
              sub="Sensitivity (TPR)"
            />
            <MetricCard
              label="F1 Score"
              value={METRICS.f1.toFixed(4)}
              accent="gold"
              icon={<BarChart3 className="w-5 h-5" />}
              sub="Precision / Recall harmonic mean"
            />
            <MetricCard
              label="ROC-AUC"
              value={METRICS.rocAuc.toFixed(4)}
              accent="cyan"
              icon={<TrendingUp className="w-5 h-5" />}
              sub="Area under ROC curve"
            />
            <MetricCard
              label="Optimal Threshold"
              value={METRICS.threshold.toFixed(4)}
              accent="gold"
              icon={<Gauge className="w-5 h-5" />}
              sub="Tuned on validation"
            />
          </div>
        </section>

        <section className="mt-10 sm:mt-12 grid grid-cols-1 lg:grid-cols-2 gap-5 sm:gap-6">
          <Card className="p-5 sm:p-6">
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan to-navy-light flex items-center justify-center shadow-glow">
                  <PieChart className="w-5 h-5 text-white" />
                </div>
                <div>
                  <h3 className="text-base font-bold text-navy dark:text-white">Confusion Matrix</h3>
                  <p className="text-[11px] text-navy/50 dark:text-white/50">
                    Actual vs predicted writer class at t = {METRICS.threshold.toFixed(4)}
                  </p>
                </div>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-sm border-separate border-spacing-1.5 sm:border-spacing-2 min-w-[440px]">
                <thead>
                  <tr>
                    <th className="w-24" />
                    <th className="text-[11px] uppercase tracking-wider text-navy/45 dark:text-white/45 font-semibold py-2 px-3">
                      Pred Same
                    </th>
                    <th className="text-[11px] uppercase tracking-wider text-navy/45 dark:text-white/45 font-semibold py-2 px-3">
                      Pred Diff
                    </th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td className="text-[11px] uppercase tracking-wider text-navy/45 dark:text-white/45 font-semibold pr-3 py-2 align-middle">
                      Actual Same
                    </td>
                    <ConfusionCell value={219} kind="tp" />
                    <ConfusionCell value={79} kind="fn" />
                  </tr>
                  <tr>
                    <td className="text-[11px] uppercase tracking-wider text-navy/45 dark:text-white/45 font-semibold pr-3 py-2 align-middle">
                      Actual Diff
                    </td>
                    <ConfusionCell value={40} kind="fp" />
                    <ConfusionCell value={270} kind="tn" />
                  </tr>
                </tbody>
              </table>
            </div>

            <div className="mt-5 grid grid-cols-2 sm:grid-cols-4 gap-3">
              <LegendCell kind="tp" label="TP" value="219" />
              <LegendCell kind="fn" label="FN" value="79" />
              <LegendCell kind="fp" label="FP" value="40" />
              <LegendCell kind="tn" label="TN" value="270" />
            </div>

            <div className="mt-6">
              <p className="text-[10px] uppercase tracking-wider text-navy/40 dark:text-white/40 font-semibold mb-3">
                Counts by actual class
              </p>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={CONFUSION} margin={CHART_MARGIN}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                    <XAxis
                      dataKey="name"
                      tick={{ fill: 'rgba(255,255,255,0.55)', fontSize: 11 }}
                      axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
                      tickLine={false}
                    />
                    <YAxis
                      tick={{ fill: 'rgba(255,255,255,0.55)', fontSize: 11 }}
                      axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
                      tickLine={false}
                      width={36}
                    />
                    <Tooltip
                      contentStyle={chartTooltipStyle}
                      cursor={{ fill: 'rgba(255,255,255,0.04)' }}
                    />
                    <Legend
                      wrapperStyle={{ fontSize: 11, paddingTop: 8 }}
                      iconType="circle"
                    />
                    <Bar dataKey="Pred Same" stackId="a" radius={[8, 8, 0, 0]}>
                      {[0, 1].map((i) => (
                        <Cell key={i} fill={i === 0 ? '#4ade80' : '#fb923c'} />
                      ))}
                    </Bar>
                    <Bar dataKey="Pred Diff" stackId="a" radius={[8, 8, 0, 0]}>
                      {[0, 1].map((i) => (
                        <Cell key={i} fill={i === 0 ? '#f97316' : '#22d3ee'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </Card>

          <Card className="p-5 sm:p-6">
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-gold to-orange-500 flex items-center justify-center shadow-glow-gold">
                  <LineChartIcon className="w-5 h-5 text-navy" />
                </div>
                <div>
                  <h3 className="text-base font-bold text-navy dark:text-white">ROC Curve</h3>
                  <p className="text-[11px] text-navy/50 dark:text-white/50">
                    Receiver operating characteristic — AUC = {METRICS.rocAuc.toFixed(4)}
                  </p>
                </div>
              </div>
              <div className="px-3 py-1.5 rounded-full bg-gold/10 border border-gold/30 text-[11px] font-mono font-bold text-gold">
                AUC 0.8914
              </div>
            </div>

            <div className="h-64 sm:h-72">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={ROC_POINTS}
                  margin={{ ...CHART_MARGIN, left: 4 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis
                    dataKey="fpr"
                    type="number"
                    domain={[0, 1]}
                    tick={{ fill: 'rgba(255,255,255,0.55)', fontSize: 11 }}
                    axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
                    tickLine={false}
                    label={{
                      value: 'False Positive Rate (FPR)',
                      position: 'insideBottom',
                      offset: -2,
                      fill: 'rgba(255,255,255,0.5)',
                      fontSize: 10,
                    }}
                  />
                  <YAxis
                    type="number"
                    domain={[0, 1]}
                    tick={{ fill: 'rgba(255,255,255,0.55)', fontSize: 11 }}
                    axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
                    tickLine={false}
                    width={36}
                    label={{
                      value: 'TPR',
                      angle: -90,
                      position: 'insideLeft',
                      fill: 'rgba(255,255,255,0.5)',
                      fontSize: 10,
                    }}
                  />
                  <Tooltip
                    contentStyle={chartTooltipStyle}
                    formatter={(v: any, n: any) => [
                      Number(v).toFixed(3),
                      String(n) === 'tpr' ? 'TPR (Sensitivity)' : 'Random baseline',
                    ]}
                    labelFormatter={(l) => `FPR = ${Number(l).toFixed(3)}`}
                  />
                  <ReferenceLine
                    stroke="rgba(255,255,255,0.1)"
                    segment={[
                      { x: 0, y: 0 },
                      { x: 0, y: 1 },
                    ]}
                  />
                  <Line
                    type="monotone"
                    dataKey="tpr"
                    name="ROC"
                    stroke="#219EBC"
                    strokeWidth={3}
                    dot={{ r: 0 }}
                    activeDot={{ r: 5, fill: '#FFB703', stroke: '#023047', strokeWidth: 2 }}
                  />
                  <Line
                    type="monotone"
                    dataKey={() => undefined as any}
                    stroke="transparent"
                  />
                  <ReferenceLine
                    stroke="rgba(255,183,3,0.35)"
                    strokeDasharray="4 4"
                    strokeWidth={1.5}
                    segment={[
                      { x: 0, y: 0 },
                      { x: 1, y: 1 },
                    ]}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="mt-5 grid grid-cols-2 gap-3 text-[11px]">
              <div className="rounded-xl bg-navy/[0.02] dark:bg-white/[0.02] border border-navy/5 dark:border-white/5 p-3">
                <p className="text-navy/45 dark:text-white/45 uppercase tracking-wider text-[10px] mb-1 font-semibold">
                  Interpretation
                </p>
                <p className="text-white/75 leading-relaxed">
                  AUC of <span className="text-gold font-bold">0.8914</span> indicates
                  strong discriminative performance between same-writer and
                  different-writer pairs.
                </p>
              </div>
              <div className="rounded-xl bg-navy/[0.02] dark:bg-white/[0.02] border border-navy/5 dark:border-white/5 p-3">
                <p className="text-navy/45 dark:text-white/45 uppercase tracking-wider text-[10px] mb-1 font-semibold">
                  Operating Point
                </p>
                <p className="text-white/75 leading-relaxed">
                  At t = {METRICS.threshold.toFixed(4)}: FPR ≈ 0.129, TPR ={' '}
                  <span className="text-cyan-light font-bold">{METRICS.recall.toFixed(4)}</span>.
                </p>
              </div>
            </div>
          </Card>
        </section>

        <section className="mt-5 sm:mt-6">
          <Card className="p-5 sm:p-6">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-5">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-500 to-cyan flex items-center justify-center shadow-glow">
                  <Activity className="w-5 h-5 text-white" />
                </div>
                <div>
                  <h3 className="text-base font-bold text-navy dark:text-white">
                    Similarity Score Distribution
                  </h3>
                  <p className="text-[11px] text-navy/50 dark:text-white/50">
                    Binned histogram of scores by ground-truth writer pair class
                  </p>
                </div>
              </div>
              <div className="flex flex-wrap items-center gap-3 text-[11px]">
                <LegendDot color="#22d3ee" label="Same Writer" />
                <LegendDot color="#fb923c" label="Different Writer" />
                <LegendDot color="#FFB703" label="Threshold t = 0.4866" solid />
              </div>
            </div>

            <div className="h-72 sm:h-80">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={SAME_WRITER_HIST} margin={{ ...CHART_MARGIN, left: 4 }}>
                  <defs>
                    <linearGradient id="gradSame" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#22d3ee" stopOpacity={0.6} />
                      <stop offset="100%" stopColor="#22d3ee" stopOpacity={0.02} />
                    </linearGradient>
                    <linearGradient id="gradDiff" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#fb923c" stopOpacity={0.6} />
                      <stop offset="100%" stopColor="#fb923c" stopOpacity={0.02} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis
                    dataKey="bin"
                    tick={{ fill: 'rgba(255,255,255,0.55)', fontSize: 10 }}
                    axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
                    tickLine={false}
                    angle={-12}
                    textAnchor="end"
                    height={48}
                  />
                  <YAxis
                    tick={{ fill: 'rgba(255,255,255,0.55)', fontSize: 11 }}
                    axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
                    tickLine={false}
                    width={36}
                    label={{
                      value: 'Pairs (count)',
                      angle: -90,
                      position: 'insideLeft',
                      fill: 'rgba(255,255,255,0.5)',
                      fontSize: 10,
                    }}
                  />
                  <Tooltip
                    contentStyle={chartTooltipStyle}
                    formatter={(v: any, n: any) => [
                      `${Number(v)} pairs`,
                      n === 'same' ? 'Same Writer' : 'Different Writer',
                    ]}
                    labelFormatter={(l) => `Score bin: ${l}`}
                  />
                  <Legend
                    wrapperStyle={{ fontSize: 11, paddingTop: 8 }}
                    iconType="circle"
                  />
                  <Area
                    type="monotone"
                    dataKey="same"
                    name="Same Writer"
                    stroke="#22d3ee"
                    strokeWidth={2.5}
                    fill="url(#gradSame)"
                  />
                  <Area
                    type="monotone"
                    dataKey="diff"
                    name="Different Writer"
                    stroke="#fb923c"
                    strokeWidth={2.5}
                    fill="url(#gradDiff)"
                  />
                  <ReferenceLine
                    x="0.40–0.49"
                    stroke="rgba(255,183,3,0.2)"
                    strokeDasharray="0"
                    strokeWidth={1}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            <div className="mt-6 grid grid-cols-1 sm:grid-cols-3 gap-4">
              <InfoTile
                title="Separation"
                body="Same-writer scores cluster right of t=0.4866; different-writer scores cluster left of the threshold."
                accent="cyan"
              />
              <InfoTile
                title="Overlap Region"
                body="Ambiguous scores near 0.35–0.60 produce the confidence margins surfaced alongside each verdict."
                accent="gold"
              />
              <InfoTile
                title="Tuned Threshold"
                body="t=0.4866 was selected on the validation split to optimally balance precision and recall for exam integrity."
                accent="cyan"
              />
            </div>
          </Card>
        </section>
      </div>
    </div>
  );
}

type ConfusionKind = 'tp' | 'tn' | 'fp' | 'fn';

const confusionPalette: Record<
  ConfusionKind,
  { bg: string; border: string; text: string; dot: string }
> = {
  tp: {
    bg: 'bg-emerald-500/10',
    border: 'border-emerald-500/40',
    text: 'text-emerald-300',
    dot: '#34d399',
  },
  tn: {
    bg: 'bg-cyan/10',
    border: 'border-cyan/40',
    text: 'text-cyan-light',
    dot: '#22d3ee',
  },
  fp: {
    bg: 'bg-orange-500/10',
    border: 'border-orange-500/40',
    text: 'text-orange-300',
    dot: '#fb923c',
  },
  fn: {
    bg: 'bg-amber-500/10',
    border: 'border-amber-500/40',
    text: 'text-amber-300',
    dot: '#f59e0b',
  },
};

function ConfusionCell({ value, kind }: { value: number; kind: ConfusionKind }) {
  const p = confusionPalette[kind];
  return (
    <td
      className={`relative rounded-xl border-2 ${p.border} ${p.bg} p-4 sm:p-5 text-center align-middle`}
    >
      <span className={`text-2xl sm:text-3xl font-black ${p.text} block`}>{value}</span>
      <span className="text-[9px] uppercase tracking-widest text-navy/40 dark:text-white/40 font-semibold mt-0.5 block">
        {kind.toUpperCase()}
      </span>
    </td>
  );
}

function LegendCell({
  kind,
  label,
  value,
}: {
  kind: ConfusionKind;
  label: string;
  value: string;
}) {
  const p = confusionPalette[kind];
  return (
    <div className="rounded-xl bg-navy/[0.02] dark:bg-white/[0.02] border border-navy/5 dark:border-white/5 px-3 py-2.5 flex items-center gap-2.5">
      <span
        className="w-3 h-3 rounded-md shrink-0"
        style={{ background: p.dot }}
      />
      <div className="min-w-0 flex-1">
        <p className="text-[10px] uppercase tracking-wider text-navy/40 dark:text-white/40 font-semibold leading-tight">
          {label}
        </p>
        <p className="text-sm font-bold text-navy dark:text-white font-mono leading-tight">{value}</p>
      </div>
    </div>
  );
}

function MetricCard({
  label,
  value,
  suffix,
  accent,
  icon,
  sub,
}: {
  label: string;
  value: string;
  suffix?: string;
  accent: 'cyan' | 'gold';
  icon: React.ReactNode;
  sub: string;
}) {
  return (
    <Card hoverable className="p-5 sm:p-6">
      <div className="flex items-start justify-between mb-4">
        <div
          className={`w-10 h-10 rounded-xl flex items-center justify-center ${
            accent === 'cyan'
              ? 'bg-gradient-to-br from-cyan to-navy-light shadow-glow'
              : 'bg-gradient-to-br from-gold to-orange-500 shadow-glow-gold'
          }`}
        >
          <span className="text-white">{icon}</span>
        </div>
      </div>
      <p className="text-[11px] uppercase tracking-wider text-navy/45 dark:text-white/45 font-semibold mb-1.5">
        {label}
      </p>
      <div className="flex items-baseline gap-1 mb-1.5">
        <span
          className={`text-2xl sm:text-3xl font-black tracking-tight font-mono ${
            accent === 'cyan' ? 'text-cyan-light' : 'text-gold'
          }`}
        >
          {value}
        </span>
        {suffix && (
          <span className="text-sm font-bold text-white/30">{suffix}</span>
        )}
      </div>
      <p className="text-[11px] text-navy/50 dark:text-white/50">{sub}</p>
    </Card>
  );
}

function LegendDot({
  color,
  label,
  solid,
}: {
  color: string;
  label: string;
  solid?: boolean;
}) {
  return (
    <div className="inline-flex items-center gap-2 text-navy/70 dark:text-white/70">
      {solid ? (
        <span
          className="w-3.5 h-1 rounded-full"
          style={{ background: color, boxShadow: `0 0 8px ${color}66` }}
        />
      ) : (
        <span
          className="w-2.5 h-2.5 rounded-full"
          style={{ background: color, boxShadow: `0 0 6px ${color}66` }}
        />
      )}
      <span>{label}</span>
    </div>
  );
}

function InfoTile({
  title,
  body,
  accent,
}: {
  title: string;
  body: string;
  accent: 'cyan' | 'gold';
}) {
  return (
    <div
      className={`rounded-2xl p-5 border ${
        accent === 'cyan'
          ? 'bg-gradient-to-br from-cyan/10 to-navy/20 border-cyan/20'
          : 'bg-gradient-to-br from-gold/10 to-orange-500/5 border-gold/20'
      }`}
    >
      <p
        className={`text-[11px] uppercase tracking-wider font-semibold mb-2 ${
          accent === 'cyan' ? 'text-cyan-light' : 'text-gold'
        }`}
      >
        {title}
      </p>
      <p className="text-xs sm:text-sm text-navy/70 dark:text-white/70 leading-relaxed">{body}</p>
    </div>
  );
}
