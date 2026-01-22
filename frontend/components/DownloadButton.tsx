import React from "react";

export default function DownloadButton({
  disabled,
  onDownloadFinal,
  onDownloadValidated
}: {
  disabled: boolean;
  onDownloadFinal: () => void;
  onDownloadValidated: () => void;
}) {
  return (
    <div className="btnGroup">
      <button className="btnGhost" disabled={disabled} onClick={onDownloadFinal}>
        Download Final
      </button>
      <button className="btnGhost" disabled={disabled} onClick={onDownloadValidated}>
        Download Validated
      </button>
    </div>
  );
}
