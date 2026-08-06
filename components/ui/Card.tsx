import * as React from 'react';

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
  hoverable?: boolean;
  glass?: boolean;
}

export default function Card({
  children,
  hoverable = false,
  glass = true,
  className = '',
  ...props
}: CardProps) {
  return (
    <div
      className={`
        ${glass ? 'glass dark:glass' : ''}
        rounded-2xl
        ${hoverable ? 'transition-all duration-500 hover:-translate-y-0.5 hover:shadow-glow hover:border-cyan/20' : ''}
        ${className}
      `}
      {...props}
    >
      {children}
    </div>
  );
}

interface CardHeaderProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
}

Card.Header = function CardHeader({
  children,
  className = '',
  ...props
}: CardHeaderProps) {
  return (
    <div
      className={`flex flex-col space-y-1.5 p-6 pb-4 ${className}`}
      {...props}
    >
      {children}
    </div>
  );
};

interface CardTitleProps extends React.HTMLAttributes<HTMLHeadingElement> {
  children: React.ReactNode;
}

Card.Title = function CardTitle({
  children,
  className = '',
  ...props
}: CardTitleProps) {
  return (
    <h3
      className={`text-lg font-bold leading-none tracking-tight text-navy dark:text-white ${className}`}
      {...props}
    >
      {children}
    </h3>
  );
};

interface CardDescriptionProps
  extends React.HTMLAttributes<HTMLParagraphElement> {
  children: React.ReactNode;
}

Card.Description = function CardDescription({
  children,
  className = '',
  ...props
}: CardDescriptionProps) {
  return (
    <p
      className={`text-sm text-navy/50 dark:text-white/50 ${className}`}
      {...props}
    >
      {children}
    </p>
  );
};

interface CardContentProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
}

Card.Content = function CardContent({
  children,
  className = '',
  ...props
}: CardContentProps) {
  return (
    <div className={`p-6 pt-0 ${className}`} {...props}>
      {children}
    </div>
  );
};

interface CardFooterProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
}

Card.Footer = function CardFooter({
  children,
  className = '',
  ...props
}: CardFooterProps) {
  return (
    <div
      className={`flex items-center p-6 pt-0 border-t border-navy/5 dark:border-white/5 mt-4 ${className}`}
      {...props}
    >
      {children}
    </div>
  );
};
