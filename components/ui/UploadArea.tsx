'use client';

import * as React from 'react';
import { Upload, Image as ImageIcon, X, FileImage, AlertCircle } from 'lucide-react';

interface UploadAreaProps {
  label: string;
  index: number;
  imageData: ImageData | null;
  previewUrl: string | null;
  fileName: string | null;
  onFile: (file: File) => void;
  onClear: () => void;
  error?: string | null;
}

const ACCEPTED_TYPES = ['image/png', 'image/jpg', 'image/jpeg'];

export default function UploadArea({
  label,
  index,
  previewUrl,
  fileName,
  onFile,
  onClear,
  error = null,
}: UploadAreaProps) {
  const [isDragging, setIsDragging] = React.useState(false);
  const inputRef = React.useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      validateAndProcess(files[0]);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      validateAndProcess(files[0]);
    }
  };

  const validateAndProcess = (file: File) => {
    if (!ACCEPTED_TYPES.includes(file.type)) {
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      return;
    }
    onFile(file);
  };

  const handleClick = () => {
    if (!previewUrl) {
      inputRef.current?.click();
    }
  };

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-cyan to-navy flex items-center justify-center text-xs font-bold text-white shadow-glow">
            {index}
          </div>
          <h3 className="text-sm font-semibold text-navy dark:text-white tracking-wide">
            {label}
          </h3>
        </div>
        {previewUrl && (
          <button
            onClick={onClear}
            className="w-8 h-8 rounded-lg bg-navy/5 dark:bg-white/5 hover:bg-red-500/20 hover:text-red-400 text-navy/60 dark:text-white/60 flex items-center justify-center transition-all duration-200"
            aria-label="Clear image"
          >
            <X className="w-4 h-4" />
          </button>
        )}
      </div>

      <div
        onClick={handleClick}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`
          relative aspect-[4/3] w-full rounded-2xl border-2 border-dashed
          cursor-pointer transition-all duration-300 overflow-hidden
          ${previewUrl
            ? 'border-cyan/40 bg-cyan/5'
            : isDragging
              ? 'border-gold bg-gold/5 scale-[1.01]'
              : error
                ? 'border-red-500/50 bg-red-500/5 animate-pulse'
                : 'border-navy/15 dark:border-white/15 hover:border-cyan/50 hover:bg-navy/[0.02] dark:hover:bg-white/[0.02] bg-navy/[0.01] dark:bg-white/[0.01]'
          }
        `}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".png,.jpg,.jpeg,image/png,image/jpeg"
          onChange={handleChange}
          className="hidden"
        />

        {previewUrl ? (
          <div className="w-full h-full group relative">
            <img
              src={previewUrl}
              alt={`${label} preview`}
              className="w-full h-full object-contain bg-gradient-to-br from-slate-100 dark:from-dark-surface/50 to-slate-200 dark:to-navy/30 p-2 transition-transform duration-500 group-hover:scale-[1.02]"
              draggable={false}
            />
            <div className="absolute inset-0 bg-gradient-to-t from-dark-bg/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-end p-4 pointer-events-none">
              <p className="text-xs text-white/80 truncate w-full">{fileName}</p>
            </div>
            <div className="absolute top-3 left-3 px-2.5 py-1 rounded-md bg-green-500/20 border border-green-500/40 text-[10px] font-medium text-green-400 flex items-center gap-1">
              <div className="w-1.5 h-1.5 rounded-full bg-green-400" />
              Ready
            </div>
          </div>
        ) : (
          <div className="absolute inset-0 flex flex-col items-center justify-center p-6 text-center">
            <div
              className={`
                w-16 h-16 rounded-2xl flex items-center justify-center mb-4 transition-all duration-300
                ${isDragging
                  ? 'bg-gold/20 scale-110 shadow-glow-gold'
                  : error
                    ? 'bg-red-500/10'
                    : 'bg-cyan/10 group-hover:bg-cyan/20'
                }
              `}
            >
              {error ? (
                <AlertCircle className="w-8 h-8 text-red-400" />
              ) : isDragging ? (
                <FileImage className="w-8 h-8 text-gold" />
              ) : (
                <Upload className="w-8 h-8 text-cyan" />
              )}
            </div>

            <p className="text-sm font-medium text-navy dark:text-white mb-1.5">
              {isDragging ? 'Drop image here' : 'Click or drag & drop'}
            </p>
            <p className="text-xs text-navy/40 dark:text-white/40 mb-3">
              PNG, JPG, JPEG up to 10MB
            </p>

            <div className="flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-navy/5 dark:bg-white/5 border border-navy/5 dark:border-white/5">
              <ImageIcon className="w-3 h-3 text-cyan" />
              <span className="text-[10px] text-navy/50 dark:text-white/50 font-mono tracking-wide">
                112 x 896 grayscale
              </span>
            </div>

            {error && (
              <div className="mt-4 px-4 py-2 rounded-xl bg-red-500/10 border border-red-500/30 text-xs text-red-400 w-full max-w-xs">
                {error}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
