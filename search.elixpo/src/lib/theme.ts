export const colors = {
  bg: {
    deep: '#0a0c14',
    card: 'rgba(12, 16, 28, 0.8)',
    cardGlass: 'rgba(255,255,255,0.05)',
    cardGlassHover: 'rgba(255,255,255,0.08)',
    overlay: 'rgba(10, 12, 20, 0.95)',
    surface: '#111',
    elevated: '#1a1a1a',
  },
  lime: { main: '#444ce7', light: '#5558e8', dim: 'rgba(68, 76, 231, 0.15)' },
  sage: { main: '#6ea8fe', dim: 'rgba(110, 168, 254, 0.15)' },
  honey: { main: '#818cf8', dim: 'rgba(129, 140, 248, 0.15)' },
  lavender: { main: '#c4b5fd', light: '#d8b4fe' },
} as const;

export const gradients = {
  accent: 'linear-gradient(90deg, #444ce7, #6ea8fe, #818cf8)',
  textHero: 'linear-gradient(135deg, #f5f5f4 0%, #444ce7 30%, #6ea8fe 60%, #818cf8 100%)',
} as const;
