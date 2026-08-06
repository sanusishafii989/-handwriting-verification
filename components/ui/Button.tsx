import * as React from 'react';
import Link from 'next/link';
import { Loader2 } from 'lucide-react';

type Variant = 'primary' | 'secondary' | 'outline' | 'ghost' | 'danger' | 'gold';
type Size = 'sm' | 'md' | 'lg' | 'xl';

interface ButtonBaseProps {
  variant?: Variant;
  size?: Size;
  isLoading?: boolean;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
  className?: string;
  children?: React.ReactNode;
}

interface ButtonAsButton
  extends ButtonBaseProps,
    Omit<React.ButtonHTMLAttributes<HTMLButtonElement>, keyof ButtonBaseProps> {
  href?: undefined;
}

interface ButtonAsLink
  extends ButtonBaseProps,
    Omit<React.AnchorHTMLAttributes<HTMLAnchorElement>, keyof ButtonBaseProps> {
  href: string;
}

type ButtonProps = ButtonAsButton | ButtonAsLink;

const variantClasses: Record<Variant, string> = {
  primary:
    'bg-gradient-to-r from-cyan to-navy-light text-white hover:from-cyan-light hover:to-cyan shadow-glow hover:shadow-glow-gold',
  secondary:
    'bg-navy/10 text-navy dark:bg-white/10 dark:text-white hover:bg-navy/15 dark:hover:bg-white/15 border border-navy/10 dark:border-white/10',
  outline:
    'bg-transparent text-cyan border border-cyan/50 hover:bg-cyan/10 hover:border-cyan',
  ghost:
    'bg-transparent text-navy/70 dark:text-white/70 hover:bg-navy/5 dark:hover:bg-white/5 hover:text-navy dark:hover:text-white',
  danger:
    'bg-gradient-to-r from-red-500 to-red-600 text-white hover:from-red-600 hover:to-red-700 shadow-lg',
  gold:
    'bg-gradient-to-r from-gold to-orange-500 text-navy hover:from-yellow-400 hover:to-gold shadow-glow-gold font-semibold',
};

const sizeClasses: Record<Size, string> = {
  sm: 'h-8 px-3 text-xs gap-1.5 rounded-lg',
  md: 'h-10 px-5 text-sm gap-2 rounded-xl',
  lg: 'h-12 px-7 text-base gap-2.5 rounded-2xl',
  xl: 'h-14 px-9 text-base gap-3 rounded-2xl',
};

export default function Button(props: ButtonProps) {
  const {
    variant = 'primary',
    size = 'md',
    isLoading = false,
    leftIcon,
    rightIcon,
    className = '',
    children,
  } = props;

  const sharedClasses = `
    relative inline-flex items-center justify-center
    font-medium transition-all duration-300
    focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan/50
    disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:shadow-none
    active:scale-[0.98]
    ${variantClasses[variant]}
    ${sizeClasses[size]}
    ${className}
  `;

  const content = isLoading ? (
    <Loader2 className="w-4 h-4 animate-spin" />
  ) : (
    <>
      {leftIcon && <span className="shrink-0">{leftIcon}</span>}
      {children}
      {rightIcon && <span className="shrink-0">{rightIcon}</span>}
    </>
  );

  if ('href' in props && props.href) {
    const hrefValue = props.href;
    const isExternal = /^https?:\/\//.test(hrefValue);
    const anchorDisabled = isLoading || props['aria-disabled'];
    const baseRest: typeof props = props;
    const {
      href: _h,
      variant: _v,
      size: _s,
      isLoading: _l,
      leftIcon: _li,
      rightIcon: _ri,
      className: _c,
      children: _ch,
      ...rest
    } = baseRest as any;
    if (isExternal) {
      return (
        <a
          href={hrefValue}
          target="_blank"
          rel="noreferrer noopener"
          aria-disabled={anchorDisabled}
          className={sharedClasses + (anchorDisabled ? ' pointer-events-none' : '')}
          {...rest}
        >
          {content}
        </a>
      );
    }
    return (
      <Link
        href={hrefValue}
        aria-disabled={anchorDisabled}
        className={sharedClasses + (anchorDisabled ? ' pointer-events-none' : '')}
        {...rest}
      >
        {content}
      </Link>
    );
  }

  const { variant: _v, size: _s, isLoading: _l, leftIcon: _li, rightIcon: _ri, className: _c, children: _ch, ...btnRest } = props as ButtonAsButton;
  return (
    <button disabled={btnRest.disabled || isLoading} className={sharedClasses} {...btnRest}>
      {content}
    </button>
  );
}
