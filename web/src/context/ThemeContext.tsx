import { createContext, useContext } from 'react';

interface ThemeConfig {
  bg: string;
  card: string;
  text: string;
  accent: string;
  button: string;
}

const darkTheme: ThemeConfig = {
  bg: 'bg-gradient-to-br from-slate-900 via-gray-900 to-black',
  card: 'bg-white/10 border-white/20',
  text: 'text-gray-50',
  accent: 'bg-blue-500 hover:bg-blue-600 text-white',
  button: 'bg-gray-700 hover:bg-gray-600 text-gray-50',
};

interface ThemeContextType {
  config: ThemeConfig;
}

const ThemeContext = createContext<ThemeContextType | null>(null);

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  return (
    <ThemeContext.Provider value={{ config: darkTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) throw new Error('useTheme must be used within ThemeProvider');
  return context;
}
