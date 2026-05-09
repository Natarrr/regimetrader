"use client";

import type { LaureateRegime } from "@/lib/types";

const REGIME_STYLE: Record<LaureateRegime, { border: string; text: string; label: string }> = {
  BULL:       { border: "border-green",  text: "text-green",  label: "BULL MARKET" },
  OVERHEATED: { border: "border-yellow", text: "text-yellow", label: "OVERHEATED" },
  FRAGILE:    { border: "border-orange-400", text: "text-orange-400", label: "FRAGILE" },
  CRASH:      { border: "border-red",    text: "text-red",    label: "CRASH / BEAR" },
};

interface Props {
  symbol: string;
  regime: LaureateRegime;
  hmm_label: string;
  monetary_regime: string;
  volatility_regime: string;
  position_scale: number;
  is_uncertain: boolean;
}

export function RegimeCard({
  symbol,
  regime,
  hmm_label,
  monetary_regime,
  volatility_regime,
  position_scale,
  is_uncertain,
}: Props) {
  const styles = REGIME_STYLE[regime];

  return (
    <div className={`bg-surface border ${styles.border} p-4`}>
      <div className="flex justify-between items-start mb-3">
        <span className="text-[10px] text-dim tracking-widest">LAUREATE REGIME</span>
        {is_uncertain && (
          <span className="text-[9px] border border-yellow text-yellow px-1">UNCERTAIN</span>
        )}
      </div>
      <div className={`text-2xl font-bold tracking-widest mb-2 ${styles.text}`}>
        {symbol} — {styles.label}
      </div>
      <div className="grid grid-cols-3 gap-2 text-[10px] text-dim mt-3">
        <Stat label="HMM STATE"   value={hmm_label} />
        <Stat label="MONETARY"    value={monetary_regime} />
        <Stat label="VOLATILITY"  value={volatility_regime} />
        <Stat
          label="POSITION SCALE"
          value={`${(position_scale * 100).toFixed(0)}%`}
          highlight={position_scale < 0.8}
        />
      </div>
    </div>
  );
}

function Stat({ label, value, highlight = false }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div>
      <div className="text-[9px] text-dim mb-0.5">{label}</div>
      <div className={`font-mono text-xs ${highlight ? "text-yellow" : "text-text"}`}>{value}</div>
    </div>
  );
}
