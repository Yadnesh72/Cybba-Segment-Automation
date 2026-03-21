import React from "react";

import ThemeToggle from "./ThemeToggle";
import { motion, AnimatePresence } from "framer-motion";
import { useTheme } from "../src/Theme"; // adjust if your folders differ
type MenuItem = {
  id: string;
  label: string;
  icon?: React.ReactNode;
};

export default function SideMenu({
  active,
  onSelect,
  disabledIds = [],
}: {
  active: string;
  onSelect: (id: string) => void;
  disabledIds?: string[];
}) {
  const items: MenuItem[] = [
    { id: "segments", label: "Segments" },
    { id: "comparison", label: "Comparison" },
    { id: "analytics", label: "Analytics" },
    { id: "cybba", label: "Cybba segments" }, 
    { id: "suggestions", label: "Suggestions" },// ✅ add this

  ];
  const { theme, toggleTheme } = useTheme();
  const isDisabled = (id: string) => disabledIds.includes(id);
  return (
    <aside className="sideNav sideNavOpen">
     <div className="sideNavTop">
        <div className="sideNavBrand">
          <div className="sideNavTitle">Menu</div>
        </div>

        <ThemeToggle />
      </div>

      <nav className="sideNavItems">
        {items.map((it) => {
          const isActive = active === it.id;
          const disabled = isDisabled(it.id);

          return (
            <button
                key={it.id}
                className={`sideNavItem ${isActive ? "sideNavItemActive" : ""} ${disabled ? "sideNavItemDisabled" : ""}`}
                onClick={() => {
                  if (disabled) return;
                  onSelect(it.id);
                }}
                title={disabled ? "Available after generation completes" : it.label}
                type="button"
                disabled={disabled}
                aria-disabled={disabled}
              >
              <span className="sideNavDot" />
              <span className="sideNavLabel">{it.label}</span>
            </button>
          );
        })}
      </nav>

      <div className="sideNavBottom">
        <div className="sideNavHint">
          Pages here will use <b>last run</b> data for charts & comparisons.
        </div>
      </div>
    </aside>
  );
}