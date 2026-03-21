import React, { createContext, useContext, useEffect, useMemo, useState } from "react";

export type ThemeMode = "dark" | "light";

type ThemeCtx = {
  theme: ThemeMode;
  setTheme: (t: ThemeMode) => void;
  toggleTheme: () => void;
};

const ThemeContext = createContext<ThemeCtx | null>(null);

function getInitialTheme(): ThemeMode {
  const saved = localStorage.getItem("theme");
  if (saved === "light" || saved === "dark") return saved;

  // default to dark (matches your current look)
  return "dark";
}

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setTheme] = useState<ThemeMode>(getInitialTheme);

useEffect(() => {
  localStorage.setItem("theme", theme);

  const root = document.documentElement;

  // smooth transition for theme swap
  root.classList.add("theme-anim");
  window.setTimeout(() => root.classList.remove("theme-anim"), 350);

  root.setAttribute("data-theme", theme);
}, [theme]);

  const value = useMemo(
    () => ({
      theme,
      setTheme,
      toggleTheme: () => setTheme((t) => (t === "dark" ? "light" : "dark")),
    }),
    [theme]
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme() {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error("useTheme must be used inside ThemeProvider");
  return ctx;
}