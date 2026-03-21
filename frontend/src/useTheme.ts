import { useEffect, useMemo, useState } from "react";

export type ThemeMode = "dark" | "light";

const STORAGE_KEY = "cybba_theme";

function getInitialTheme(): ThemeMode {
  // 1) persisted
  const saved = (typeof window !== "undefined" && localStorage.getItem(STORAGE_KEY)) as ThemeMode | null;
  if (saved === "dark" || saved === "light") return saved;

  // 2) system preference
  if (typeof window !== "undefined" && window.matchMedia) {
    return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  }

  return "dark";
}

export function useTheme() {
  const [theme, setTheme] = useState<ThemeMode>(() => getInitialTheme());

  useEffect(() => {
    // Apply to <html> so CSS can target [data-theme="light"]
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem(STORAGE_KEY, theme);
  }, [theme]);

  const api = useMemo(() => {
    return {
      theme,
      isLight: theme === "light",
      toggle: () => setTheme((t) => (t === "light" ? "dark" : "light")),
      setTheme,
    };
  }, [theme]);

  return api;
}