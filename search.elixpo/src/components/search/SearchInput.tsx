'use client';

import { useState, useRef, useCallback, useEffect, KeyboardEvent } from 'react';
import {
  ArrowUp, Plus, Globe, GraduationCap, Users,
  Upload, Search, Zap, X,
} from 'lucide-react';

interface SearchInputProps {
  onSend: (query: string) => void;
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

export default function SearchInput({ onSend, disabled, showPills }: SearchInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [plusOpen, setPlusOpen] = useState(false);
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const [activeSources, setActiveSources] = useState<Set<SourceType>>(new Set(['web']));
  const plusRef = useRef<HTMLDivElement>(null);

  const handleSend = useCallback(() => {
    const value = textareaRef.current?.value.trim();
    if (!value || disabled) return;
    onSend(value);
    if (textareaRef.current) {
      textareaRef.current.value = '';
      textareaRef.current.style.height = 'auto';
    }
  }, [onSend, disabled]);

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

  return (
    <div className="relative bg-transparent p-[5px] w-full flex flex-col items-center gap-3">
      <div className="max-w-[768px] w-full mx-auto flex flex-col gap-2 bg-[#2a2b2d] rounded-3xl px-4 py-3 border border-[#333] shadow-[0px_-10px_20px_#111] focus-within:border-[#444] transition-colors">
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

        {/* Bottom bar */}
        <div className="flex items-center justify-between text-[#888]">
          {/* Left: + button */}
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
                <button className="w-full flex items-center gap-2.5 px-4 py-2.5 text-sm text-[#ccc] hover:text-white hover:bg-[#2a2b2d] transition-colors">
                  <Upload size={15} />
                  Upload files or images
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

                <button className="w-full flex items-center gap-2.5 px-4 py-2.5 text-sm text-[#ccc] hover:text-white hover:bg-[#2a2b2d] transition-colors">
                  <Zap size={15} />
                  Deep research
                </button>
              </div>
            )}
          </div>

          {/* Right: send */}
          <div className="flex items-center gap-2">
            <button
              onClick={handleSend}
              disabled={disabled}
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
