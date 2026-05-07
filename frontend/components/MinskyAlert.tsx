"use client";

import type { MinskyStatusOut } from "@/lib/types";

const LEVEL_STYLES: Record<string, string> = {
  CLEAR: "border-green text-green",
  WATCH: "border-yellow text-yellow",
  WARNING: "border-orange-400 text-orange-400",
  CRITICAL: "border-red text-red animate-pulse",
};

const LEVEL_BG: Record<string, string> = {
  CLEAR: "bg-green/5",
  WATCH: "bg-yellow/5",
  WARNING: "bg-orange-400/5",
  CRITICAL: "bg-red/10",
};

interface Props {
  data: MinskyStatusOut | null;
  loading?: boolean;
}

export function MinskyAlert({ data, loading }: Props) {
  if (loading) {
    return (
      <div className="w-full border border-border bg-surface px-4 py-2 text-dim text-xs">
        MINSKY MOMENT — LOADING...
      </div>
    );
  }

  if (!data) return null;

  const styles = LEVEL_STYLES[data.alert_level] ?? LEVEL_STYLES.CLEAR;
  const bg = LEVEL_BG[data.alert_level] ?? "";

  return (
    <div className={`w-full border ${styles} ${bg} px-4 py-2 flex items-center justify-between text-xs`}>
      <span className="font-bold tracking-widest">
        MINSKY STATUS [{data.alert_level}] — {data.conditions_met}/3 CONDITIONS MET
      </span>
      <div className="flex gap-3">
        <PillBadge
          label="GARCH"
          value={`${(data.garch_persistence * 100).toFixed(1)}%`}
          triggered={data.garch_persistence > 0.98}
        />
        <PillBadge
          label="CAPE"
          value={`P${data.cape_percentile.toFixed(0)}`}
          triggered={data.cape_percentile > 95}
        />
        <PillBadge
          label="YIELD"
          value={`${data.yield_spread_bps.toFixed(0)}bps`}
          triggered={data.yield_spread_bps < 0}
        />
      </div>
    </div>
  );
}

function PillBadge({
  label,
  value,
  triggered,
}: {
  label: string;
  value: string;
  triggered: boolean;
}) {
  return (
    <span
      className={`px-2 py-0.5 border rounded text-[10px] font-mono tracking-wider ${
        triggered
          ? "border-red text-red bg-red/10"
          : "border-border text-dim"
      }`}
    >
      {label} {value}
    </span>
  );
}
