'use client';

import { useState, useRef, useCallback, useEffect, KeyboardEvent } from 'react';
import {
  ArrowUp, Plus, Globe, GraduationCap, Users,
  Upload, Search, Zap, X, Image as ImageIcon, Loader2,
} from 'lucide-react';

export interface SearchPayload {
  query: string;
  images?: string[];
  deepSearch?: boolean;
}

interface SearchInputProps {
  onSend: (payload: SearchPayload) => void;
  disabled?: boolean;
  showPills?: boolean;
}

type SourceType = 'web' | 'academic' | 'social';

const SOURCES: { id: SourceType; label: string; icon: typeof Globe }[] = [
  { id: 'web', label: 'Web', icon: Globe },
  { id: 'academic', label: 'Academic', icon: GraduationCap },
  { id: 'social', label: 'Social', icon: Users },
];

const QUICK_PILLS = [
  { label: 'Local', icon: Search },
  { label: 'Finance', icon: Globe },
  { label: 'Compare', icon: Globe },
  { label: 'Shopping', icon: Globe },
  { label: 'Summarize', icon: Globe },
];

const MAX_IMAGES = 3;

async function uploadToPollinations(file: File): Promise<string> {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch('https://media.pollinations.ai/upload', {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Upload failed (${res.status}): ${text}`);
  }

  const data = await res.json();
  return data.url;
}

export default function SearchInput({ onSend, disabled, showPills }: SearchInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [plusOpen, setPlusOpen] = useState(false);
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const [activeSources, setActiveSources] = useState<Set<SourceType>>(new Set(['web']));
  const [deepSearch, setDeepSearch] = useState(false);
  const [attachedImages, setAttachedImages] = useState<Array<{ url: string; preview: string }>>([]);
  const [uploading, setUploading] = useState(false);
  const plusRef = useRef<HTMLDivElement>(null);

  const handleSend = useCallback(() => {
    const value = textareaRef.current?.value.trim();
    if ((!value && attachedImages.length === 0) || disabled || uploading) return;

    const payload: SearchPayload = { query: value || '' };
    if (attachedImages.length > 0) {
      payload.images = attachedImages.map((img) => img.url);
    }
    if (deepSearch) {
      payload.deepSearch = true;
    }

    onSend(payload);

    if (textareaRef.current) {
      textareaRef.current.value = '';
      textareaRef.current.style.height = 'auto';
    }
    setAttachedImages([]);
  }, [onSend, disabled, uploading, attachedImages, deepSearch]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend]
  );

  const handleInput = () => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 300) + 'px';
  };

  const toggleSource = (id: SourceType) => {
    setActiveSources((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        if (next.size > 1) next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const handleFileSelect = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    const remaining = MAX_IMAGES - attachedImages.length;
    const toUpload = Array.from(files).slice(0, remaining);
    if (toUpload.length === 0) return;

    setUploading(true);
    setPlusOpen(false);

    try {
      const results = await Promise.all(
        toUpload.map(async (file) => {
          const preview = URL.createObjectURL(file);
          try {
            const url = await uploadToPollinations(file);
            return { url, preview };
          } catch {
            URL.revokeObjectURL(preview);
            return null;
          }
        })
      );

      const successful = results.filter((r): r is { url: string; preview: string } => r !== null);
      setAttachedImages((prev) => [...prev, ...successful].slice(0, MAX_IMAGES));
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  }, [attachedImages.length]);

  const removeImage = useCallback((index: number) => {
    setAttachedImages((prev) => {
      const removed = prev[index];
      if (removed) URL.revokeObjectURL(removed.preview);
      return prev.filter((_, i) => i !== index);
    });
  }, []);

  // Close dropdowns on outside click
  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (plusRef.current && !plusRef.current.contains(e.target as Node)) {
        setPlusOpen(false);
        setSourcesOpen(false);
      }
    }
    if (plusOpen) document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [plusOpen]);

  const handlePillClick = (label: string) => {
    if (textareaRef.current) {
      textareaRef.current.value = label + ': ';
      textareaRef.current.focus();
    }
  };

  // Cleanup object URLs on unmount
  useEffect(() => {
    return () => {
      attachedImages.forEach((img) => URL.revokeObjectURL(img.preview));
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="relative bg-transparent p-[5px] w-full flex flex-col items-center gap-3">
      <div className="max-w-[768px] w-full mx-auto flex flex-col gap-2 bg-[#2a2b2d] rounded-3xl px-4 py-3 border border-[#333] shadow-[0px_-10px_20px_#111] focus-within:border-[#444] transition-colors">

        {/* Image previews */}
        {attachedImages.length > 0 && (
          <div className="flex items-center gap-2 px-1">
            {attachedImages.map((img, i) => (
              <div key={i} className="relative group w-16 h-16 rounded-lg overflow-hidden border border-[#444] shrink-0">
                <img src={img.preview} alt={`Attached ${i + 1}`} className="w-full h-full object-cover" />
                <button
                  onClick={() => removeImage(i)}
                  className="absolute top-0.5 right-0.5 bg-black/70 rounded-full p-0.5 opacity-0 group-hover:opacity-100 transition-opacity"
                >
                  <X size={12} className="text-white" />
                </button>
              </div>
            ))}
            {uploading && (
              <div className="w-16 h-16 rounded-lg border border-[#444] flex items-center justify-center bg-[#333]">
                <Loader2 size={18} className="text-[#888] animate-spin" />
              </div>
            )}
          </div>
        )}

        {/* Deep search indicator */}
        {deepSearch && (
          <div className="flex items-center gap-1.5 px-1">
            <span className="flex items-center gap-1 px-2 py-0.5 rounded-full bg-[#444ce7]/15 text-[#6ea8fe] text-xs font-medium border border-[#444ce7]/30">
              <Zap size={11} />
              Deep Research
              <button onClick={() => setDeepSearch(false)} className="ml-1 hover:text-white">
                <X size={11} />
              </button>
            </span>
          </div>
        )}

        {/* Textarea */}
        <textarea
          ref={textareaRef}
          placeholder="Ask anything..."
          className="flex-grow bg-transparent resize-none text-white text-lg placeholder-[#555] focus:outline-none px-1 min-h-[28px] max-h-[300px]"
          rows={1}
          autoComplete="off"
          spellCheck={false}
          onInput={handleInput}
          onKeyDown={handleKeyDown}
          disabled={disabled}
        />

        {/* Hidden file input */}
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          multiple
          className="hidden"
          onChange={handleFileSelect}
        />

        {/* Bottom bar */}
        <div className="flex items-center justify-between text-[#888]">
          {/* Left: + button */}
          <div className="flex items-center gap-1">
            <div className="relative" ref={plusRef}>
              <button
                onClick={() => { setPlusOpen(!plusOpen); setSourcesOpen(false); }}
                className={`p-1.5 rounded-lg transition-colors ${plusOpen ? 'bg-[#333] text-white' : 'hover:text-white hover:bg-[#333]'}`}
              >
                <Plus size={18} />
              </button>

              {/* Plus dropdown */}
              {plusOpen && (
                <div className="absolute bottom-full left-0 mb-2 w-56 bg-[#232425] border border-[#333] rounded-xl shadow-card-lg py-1.5 z-50">
                  <button
                    onClick={() => {
                      if (attachedImages.length < MAX_IMAGES) {
                        fileInputRef.current?.click();
                      }
                      setPlusOpen(false);
                    }}
                    disabled={attachedImages.length >= MAX_IMAGES}
                    className="w-full flex items-center gap-2.5 px-4 py-2.5 text-sm text-[#ccc] hover:text-white hover:bg-[#2a2b2d] transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                  >
                    <Upload size={15} />
                    Upload images
                    {attachedImages.length > 0 && (
                      <span className="text-[#666] text-xs ml-auto">{attachedImages.length}/{MAX_IMAGES}</span>
                    )}
                  </button>

                  {/* Connectors and sources */}
                  <div className="relative">
                    <button
                      onClick={() => setSourcesOpen(!sourcesOpen)}
                      className="w-full flex items-center justify-between px-4 py-2.5 text-sm text-[#ccc] hover:text-white hover:bg-[#2a2b2d] transition-colors"
                    >
                      <span className="flex items-center gap-2.5">
                        <Globe size={15} />
                        Connectors and sources
                      </span>
                      <span className="text-[#666] text-xs">&rsaquo;</span>
                    </button>

                    {/* Sources sub-menu */}
                    {sourcesOpen && (
                      <div className="absolute left-full top-0 ml-1 w-48 bg-[#232425] border border-[#333] rounded-xl shadow-card-lg py-1.5 z-50">
                        {SOURCES.map((src) => (
                          <button
                            key={src.id}
                            onClick={() => toggleSource(src.id)}
                            className="w-full flex items-center justify-between px-4 py-2.5 text-sm text-[#ccc] hover:text-white hover:bg-[#2a2b2d] transition-colors"
                          >
                            <span className="flex items-center gap-2.5">
                              <src.icon size={15} />
                              {src.label}
                            </span>
                            <div className={`w-4 h-4 rounded border flex items-center justify-center transition-colors ${
                              activeSources.has(src.id)
                                ? 'bg-[#444ce7] border-[#444ce7]'
                                : 'border-[#555]'
                            }`}>
                              {activeSources.has(src.id) && (
                                <svg width="10" height="8" viewBox="0 0 10 8" fill="none">
                                  <path d="M1 4L3.5 6.5L9 1" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                                </svg>
                              )}
                            </div>
                          </button>
                        ))}
                      </div>
                    )}
                  </div>

                  <button
                    onClick={() => { setDeepSearch(!deepSearch); setPlusOpen(false); }}
                    className={`w-full flex items-center gap-2.5 px-4 py-2.5 text-sm transition-colors ${
                      deepSearch
                        ? 'text-[#6ea8fe] bg-[#444ce7]/10'
                        : 'text-[#ccc] hover:text-white hover:bg-[#2a2b2d]'
                    }`}
                  >
                    <Zap size={15} />
                    Deep research
                    {deepSearch && (
                      <span className="ml-auto text-xs text-[#444ce7]">ON</span>
                    )}
                  </button>
                </div>
              )}
            </div>

            {/* Quick image attach button (visible when no images attached) */}
            {attachedImages.length === 0 && !uploading && (
              <button
                onClick={() => fileInputRef.current?.click()}
                className="p-1.5 rounded-lg transition-colors hover:text-white hover:bg-[#333]"
                title="Attach image"
              >
                <ImageIcon size={18} />
              </button>
            )}

            {uploading && (
              <span className="flex items-center gap-1.5 text-xs text-[#888] ml-1">
                <Loader2 size={14} className="animate-spin" />
                Uploading...
              </span>
            )}
          </div>

          {/* Right: send */}
          <div className="flex items-center gap-2">
            <button
              onClick={handleSend}
              disabled={disabled || uploading}
              className="bg-[#444ce7] hover:bg-[#5558e8] disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-xl p-2 transition-colors"
            >
              <ArrowUp size={18} />
            </button>
          </div>
        </div>
      </div>

      {/* Quick filter pills - only on landing */}
      {showPills && (
        <div className="flex items-center gap-2 max-w-[768px]">
          {QUICK_PILLS.map((pill) => (
            <button
              key={pill.label}
              onClick={() => handlePillClick(pill.label)}
              className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-full border border-[#333] text-[#ccc] text-xs font-medium hover:text-white hover:border-[#555] hover:bg-[#2a2b2d] transition-colors"
            >
              {pill.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
