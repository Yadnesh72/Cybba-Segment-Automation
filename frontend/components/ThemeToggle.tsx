import React from "react";
import { AnimatePresence, motion } from "framer-motion";
import { useTheme } from "../src/Theme"; // <-- use your actual Theme.tsx

function SunIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
      <path d="M12 17a5 5 0 1 0 0-10 5 5 0 0 0 0 10Z" stroke="currentColor" strokeWidth="2" />
      <path
        d="M12 2v2M12 20v2M4 12H2M22 12h-2M5.6 5.6 4.2 4.2M19.8 19.8 18.4 18.4M18.4 5.6 19.8 4.2M4.2 19.8 5.6 18.4"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
      <path
        d="M21 13.2A7.5 7.5 0 0 1 10.8 3 6.7 6.7 0 1 0 21 13.2Z"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export default function ThemeToggle() {
  const { theme, toggleTheme } = useTheme();
  const isLight = theme === "light";

  return (
    <button
      type="button"
      className="themeToggleBtn"
      onClick={toggleTheme}
      aria-label={isLight ? "Switch to dark mode" : "Switch to light mode"}
      title={isLight ? "Dark mode" : "Light mode"}
    >
      <span className="themeToggleTrack" />
      <motion.span
        className="themeToggleKnob"
        layout
        transition={{ type: "spring", stiffness: 520, damping: 34 }}
        style={{ display: "inline-flex", alignItems: "center", justifyContent: "center" }}
      >
        <AnimatePresence mode="wait" initial={false}>
          {isLight ? (
            <motion.span
              key="sun"
              initial={{ opacity: 0, rotate: -60, scale: 0.85 }}
              animate={{ opacity: 1, rotate: 0, scale: 1 }}
              exit={{ opacity: 0, rotate: 60, scale: 0.85 }}
              transition={{ duration: 0.18, ease: "easeOut" }}
            >
              <SunIcon />
            </motion.span>
          ) : (
            <motion.span
              key="moon"
              initial={{ opacity: 0, rotate: 60, scale: 0.85 }}
              animate={{ opacity: 1, rotate: 0, scale: 1 }}
              exit={{ opacity: 0, rotate: -60, scale: 0.85 }}
              transition={{ duration: 0.18, ease: "easeOut" }}
            >
              <MoonIcon />
            </motion.span>
          )}
        </AnimatePresence>
      </motion.span>
    </button>
  );
}