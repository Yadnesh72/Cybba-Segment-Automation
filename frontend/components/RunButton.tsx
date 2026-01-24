import React from "react";

export default function RunButton({
  onClick,
  loading,
}: {
  onClick: () => void;
  loading: boolean;
}) {
  return (
    <button
      type="button"                 // ✅ prevents form submit reload
      className="btnPrimary"
      onClick={onClick}
      disabled={loading}
    >
      <span className={`btnSpinner ${loading ? "btnSpinnerShow" : ""}`} />
      {loading ? "Running…" : "Run Pipeline"}
    </button>
  );
}
