import {
  ArrowRight,
  BarChart3,
  PenTool,
  ShieldCheck,
  Brain,
  Target,
  Activity,
  Sparkles,
  User,
  CheckCircle2,
} from 'lucide-react';
import Button from '@/components/ui/Button';
import Card from '@/components/ui/Card';

export default function HomePage() {
  return (
    <div className="relative flex-1 flex flex-col">
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -left-40 w-[600px] h-[600px] rounded-full bg-cyan/10 blur-[120px] opacity-60" />
        <div className="absolute top-1/3 -right-40 w-[500px] h-[500px] rounded-full bg-gold/10 blur-[120px] opacity-50" />
        <div className="absolute bottom-0 left-1/3 w-[400px] h-[400px] rounded-full bg-navy-light/20 blur-[100px] opacity-60" />
      </div>

      <section className="relative flex-1 flex items-center justify-center">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-16 sm:py-24 lg:py-32">
          <div className="max-w-5xl mx-auto text-center">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass mb-8 animate-fade-in">
              <Sparkles className="w-3.5 h-3.5 text-gold" />
              <span className="text-[11px] sm:text-xs font-semibold tracking-wide text-navy/80 dark:text-white/80">
                Research Project
                <span className="mx-2 text-white/20">·</span>
                <span className="text-cyan-light">Sanusi Shafii</span>
              </span>
            </div>

            <h1 className="text-3xl sm:text-5xl lg:text-6xl font-black tracking-tight text-navy dark:text-white leading-[1.08] mb-5 animate-slide-up">
              <span className="text-gradient">AI-Enabled Decision Support System</span>
              <br />
              <span className="text-navy dark:text-white">for Examination Integrity</span>
            </h1>

            <h2 className="text-xl sm:text-2xl lg:text-3xl font-bold text-white/85 tracking-tight mb-6 animate-slide-up">
              Handwriting Verification Using{' '}
              <span className="text-cyan-light">Siamese Neural Networks</span>
            </h2>

            <p className="text-sm sm:text-base lg:text-lg text-navy/60 dark:text-white/60 max-w-3xl mx-auto leading-relaxed mb-3 animate-slide-up">
              Uphold academic integrity with forensic-grade writer verification.
              Powered by an Xception-based Siamese network trained to detect whether two
              handwritten samples were produced by the same person — in milliseconds.
            </p>

            <div className="flex items-center justify-center gap-2 text-xs sm:text-sm text-navy/50 dark:text-white/50 mb-10 animate-slide-up">
              <div className="flex items-center gap-1.5">
                <User className="w-3.5 h-3.5 text-gold" />
                <span>By</span>
                <span className="font-semibold text-navy/80 dark:text-white/80">Sanusi Shafii</span>
              </div>
              <span className="text-white/20">·</span>
              <div className="flex items-center gap-1.5">
                <Brain className="w-3.5 h-3.5 text-cyan" />
                <span>Siamese · Xception · 64D Embedding</span>
              </div>
            </div>

            <div className="flex flex-col sm:flex-row items-center justify-center gap-3 sm:gap-4 animate-slide-up">
              <Button
                variant="gold"
                size="xl"
                href="/verify"
                leftIcon={<PenTool className="w-5 h-5" />}
                rightIcon={<ArrowRight className="w-5 h-5" />}
              >
                Get Started
              </Button>
              <Button
                variant="outline"
                size="xl"
                href="/dashboard"
                leftIcon={<BarChart3 className="w-5 h-5" />}
              >
                View Dashboard
              </Button>
            </div>

            <div className="mt-16 grid grid-cols-2 sm:grid-cols-4 gap-3 sm:gap-4 max-w-3xl mx-auto animate-slide-up">
              <Pill label="Accuracy" value="80.00%" accent="cyan" />
              <Pill label="ROC-AUC" value="0.8914" accent="gold" />
              <Pill label="F1 Score" value="0.7861" accent="cyan" />
              <Pill label="Threshold" value="0.4866" accent="gold" />
            </div>
          </div>
        </div>
      </section>

      <section className="relative py-16 sm:py-24">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <div className="max-w-5xl mx-auto mb-12 text-center">
            <p className="text-[11px] uppercase tracking-[0.2em] text-cyan font-semibold mb-3">
              Why This System
            </p>
            <h3 className="text-2xl sm:text-3xl lg:text-4xl font-bold text-navy dark:text-white tracking-tight">
              Engineered for Reliable Writer Matching
            </h3>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-5 sm:gap-6 max-w-6xl mx-auto">
            <Card hoverable className="p-7 sm:p-8">
              <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-cyan to-navy-light flex items-center justify-center mb-5 shadow-glow">
                <Brain className="w-6 h-6 text-white" />
              </div>
              <h4 className="text-lg font-bold text-navy dark:text-white mb-2">
                Siamese Xception Backbone
              </h4>
              <p className="text-sm text-navy/60 dark:text-white/60 leading-relaxed mb-5">
                Transfer-learned feature extractor initialized on ImageNet. Produces
                64-dimensional L2-normalized embeddings that encode stroke structure
                and writer-specific characteristics.
              </p>
              <div className="flex flex-wrap gap-2">
                <Tag>Xception</Tag>
                <Tag>64D L2</Tag>
                <Tag>Frozen Backbone</Tag>
              </div>
            </Card>

            <Card hoverable className="p-7 sm:p-8">
              <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-gold to-orange-500 flex items-center justify-center mb-5 shadow-glow-gold">
                <Target className="w-6 h-6 text-navy" />
              </div>
              <h4 className="text-lg font-bold text-navy dark:text-white mb-2">
                Optimized Decision Threshold
              </h4>
              <p className="text-sm text-navy/60 dark:text-white/60 leading-relaxed mb-5">
                Threshold <span className="font-mono text-gold">t = 0.4866</span> was
                tuned on the validation split to balance sensitivity and specificity,
                helping examiners make confident same/different-writer calls.
              </p>
              <div className="flex flex-wrap gap-2">
                <Tag>0.4866 Default</Tag>
                <Tag>Adjustable</Tag>
                <Tag>Real-time Preview</Tag>
              </div>
            </Card>

            <Card hoverable className="p-7 sm:p-8">
              <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-emerald-500 to-green-600 flex items-center justify-center mb-5 shadow-green-500/20">
                <Activity className="w-6 h-6 text-white" />
              </div>
              <h4 className="text-lg font-bold text-navy dark:text-white mb-2">
                89.14% ROC-AUC Performance
              </h4>
              <p className="text-sm text-navy/60 dark:text-white/60 leading-relaxed mb-5">
                Strong discriminative power across a range of decision thresholds.
                Precision, recall, and F1 metrics are surfaced alongside each
                verification for full auditability.
              </p>
              <div className="flex flex-wrap gap-2">
                <Tag>AUC 0.8914</Tag>
                <Tag>Precision 0.8448</Tag>
                <Tag>Recall 0.7350</Tag>
              </div>
            </Card>
          </div>
        </div>
      </section>

      <section className="relative py-16 sm:py-20">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <Card className="max-w-5xl mx-auto p-8 sm:p-12 text-center overflow-hidden relative">
            <div className="absolute inset-0 bg-gradient-to-br from-cyan/10 via-transparent to-gold/10" />
            <div className="relative">
              <ShieldCheck className="w-12 h-12 sm:w-14 sm:h-14 text-gold mx-auto mb-6" />
              <h3 className="text-2xl sm:text-3xl lg:text-4xl font-black text-navy dark:text-white tracking-tight mb-4">
                Ready to Verify Handwriting Integrity?
              </h3>
              <p className="text-sm sm:text-base text-navy/65 dark:text-white/65 max-w-2xl mx-auto mb-8 leading-relaxed">
                Start comparing handwritten samples in seconds, or explore the full
                validation metrics and model diagnostics on the dashboard.
              </p>
              <div className="flex flex-col sm:flex-row items-center justify-center gap-3 sm:gap-4">
                <Button
                  variant="gold"
                  size="lg"
                  href="/verify"
                  leftIcon={<PenTool className="w-4.5 h-4.5" />}
                  rightIcon={<ArrowRight className="w-4.5 h-4.5" />}
                >
                  Launch Verifier
                </Button>
                <Button
                  variant="outline"
                  size="lg"
                  href="/dashboard"
                  leftIcon={<BarChart3 className="w-4.5 h-4.5" />}
                >
                  Open Dashboard
                </Button>
              </div>

              <div className="mt-10 pt-8 border-t border-navy/5 dark:border-white/5 grid grid-cols-2 sm:grid-cols-3 gap-4 text-left max-w-3xl mx-auto">
                <CheckLine text="PNG / JPG / JPEG support" />
                <CheckLine text="Adjustable threshold slider" />
                <CheckLine text="Result export (JSON)" />
                <CheckLine text="Local comparison history" />
                <CheckLine text="Live inference timing" />
                <CheckLine text="Dark & light theme" />
              </div>
            </div>
          </Card>
        </div>
      </section>
    </div>
  );
}

function Pill({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent: 'cyan' | 'gold';
}) {
  return (
    <div className="glass rounded-2xl px-4 py-3 sm:px-5 sm:py-4">
      <p className="text-[10px] sm:text-[11px] uppercase tracking-wider text-navy/45 dark:text-white/45 font-semibold mb-1">
        {label}
      </p>
      <p
        className={`text-lg sm:text-xl font-black tracking-tight font-mono ${
          accent === 'cyan' ? 'text-cyan-light' : 'text-gold'
        }`}
      >
        {value}
      </p>
    </div>
  );
}

function Tag({ children }: { children: React.ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-navy/5 dark:bg-white/5 border border-navy/10 dark:border-white/10 text-[10px] font-mono text-navy/70 dark:text-white/70">
      {children}
    </span>
  );
}

function CheckLine({ text }: { text: string }) {
  return (
    <div className="flex items-center gap-2">
      <CheckCircle2 className="w-4 h-4 text-cyan shrink-0" />
      <span className="text-xs text-navy/70 dark:text-white/70">{text}</span>
    </div>
  );
}
