'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { PenTool, BarChart3, ShieldCheck, Home } from 'lucide-react';
import ThemeToggle from '@/components/theme/ThemeToggle';

export default function Header() {
  const pathname = usePathname();

  const navLinks = [
    { href: '/', label: 'Home', icon: Home },
    { href: '/verify', label: 'Verify', icon: PenTool },
    { href: '/dashboard', label: 'Dashboard', icon: BarChart3 },
  ];

  return (
    <header className="sticky top-0 z-50 w-full border-b border-navy/5 dark:border-white/5 bg-white/80 dark:bg-dark-bg/80 backdrop-blur-xl transition-colors duration-300">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between">
          <Link href="/" className="flex items-center gap-3 group">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan to-gold flex items-center justify-center shadow-glow group-hover:shadow-glow-gold transition-all duration-300">
              <ShieldCheck className="w-5 h-5 text-white" />
            </div>
            <div className="hidden sm:flex flex-col">
              <span className="text-sm font-bold text-navy dark:text-white leading-tight">
                Handwriting Verify
              </span>
              <span className="text-[10px] text-cyan-light/70 dark:text-cyan-light/70 leading-tight">
                Siamese Network System
              </span>
            </div>
          </Link>

          <nav className="flex items-center gap-1 sm:gap-2">
            {navLinks.map((link) => {
              const Icon = link.icon;
              const isActive = pathname === link.href;
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  className={`relative px-3 sm:px-4 py-2 rounded-lg text-sm font-medium transition-all duration-300 flex items-center gap-2 ${
                    isActive
                      ? 'text-gold bg-gold/10 shadow-inner'
                      : 'text-navy/70 dark:text-white/70 hover:text-navy dark:hover:text-white hover:bg-navy/5 dark:hover:bg-white/5'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span className="hidden sm:inline">{link.label}</span>
                  {isActive && (
                    <span className="absolute -bottom-[17px] left-1/2 -translate-x-1/2 w-1 h-1 rounded-full bg-gold" />
                  )}
                </Link>
              );
            })}
            <div className="w-px h-6 bg-navy/10 dark:bg-white/10 mx-1 sm:mx-2" />
            <ThemeToggle />
          </nav>
        </div>
      </div>
    </header>
  );
}
