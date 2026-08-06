'use client';

import { useEffect, useState } from 'react';
import { useTheme } from 'next-themes';
import { Sun, Moon, Loader2 } from 'lucide-react';

export default function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return (
      <button
        disabled
        className="w-9 h-9 rounded-lg bg-white/5 flex items-center justify-center"
      >
        <Loader2 className="w-4 h-4 animate-spin text-white/30" />
      </button>
    );
  }

  const isDark = theme === 'dark';

  return (
    <button
      onClick={() => setTheme(isDark ? 'light' : 'dark')}
      className="w-9 h-9 rounded-lg bg-navy/5 dark:bg-white/5 hover:bg-navy/10 dark:hover:bg-white/10 transition-all duration-300 flex items-center justify-center group relative overflow-hidden"
      aria-label="Toggle theme"
    >
      <div className="relative w-4 h-4">
        <Sun
          className={`absolute inset-0 w-4 h-4 transition-all duration-500 ${
            isDark
              ? 'opacity-0 -rotate-90 scale-50 text-gold'
              : 'opacity-100 rotate-0 scale-100 text-gold'
          }`}
        />
        <Moon
          className={`absolute inset-0 w-4 h-4 transition-all duration-500 ${
            isDark
              ? 'opacity-100 rotate-0 scale-100 text-cyan'
              : 'opacity-0 rotate-90 scale-50 text-cyan'
          }`}
        />
      </div>
      <span
        className={`absolute inset-0 rounded-lg bg-gradient-to-br from-cyan/20 to-gold/20 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none`}
      />
    </button>
  );
}
