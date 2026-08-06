export default function Footer() {
  return (
    <footer className="w-full border-t border-navy/5 dark:border-white/5 bg-slate-50/60 dark:bg-dark-bg/60 backdrop-blur-sm mt-auto transition-colors duration-300">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="space-y-3">
            <h3 className="text-sm font-bold text-navy dark:text-white tracking-wide">
              AI Handwriting Verification
            </h3>
            <p className="text-xs text-navy/50 dark:text-white/50 leading-relaxed">
              A decision support system for examination integrity powered by Siamese neural networks with Xception backbone.
            </p>
          </div>

          <div className="space-y-3">
            <h3 className="text-sm font-bold text-navy dark:text-white tracking-wide">
              Model Architecture
            </h3>
            <ul className="space-y-2 text-xs text-navy/50 dark:text-white/50">
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-cyan" />
                Backbone: Xception (ImageNet Pretrained)
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-gold" />
                Embedding: 64D L2-Normalized
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-cyan" />
                Distance: L1 (Manhattan) + Sigmoid
              </li>
            </ul>
          </div>

          <div className="space-y-3">
            <h3 className="text-sm font-bold text-navy dark:text-white tracking-wide">
              Author
            </h3>
            <div className="space-y-1 text-xs text-navy/50 dark:text-white/50">
              <p className="text-cyan font-medium">Sanusi Shafii</p>
              <p>Department of Computer Science</p>
              <p>Handwriting Verification Research</p>
            </div>
          </div>
        </div>

        <div className="mt-8 pt-6 border-t border-navy/5 dark:border-white/5 flex flex-col sm:flex-row items-center justify-between gap-4">
          <p className="text-[11px] text-navy/40 dark:text-white/40">
            &copy; {new Date().getFullYear()} Sanusi Shafii. All rights reserved.
          </p>
          <div className="flex items-center gap-2 text-[11px] text-navy/40 dark:text-white/40">
            <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            <span>Optimal Threshold: 0.4866</span>
          </div>
        </div>
      </div>
    </footer>
  );
}
