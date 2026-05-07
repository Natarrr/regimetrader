"use client";

import { RadialBarChart, RadialBar, PolarAngleAxis, ResponsiveContainer } from "recharts";

interface GaugeProps {
  label: string;
  value: number;     // 0–100 normalised
  rawValue: string;
  threshold: number; // value above which = danger
  unit?: string;
  inverted?: boolean; // if true, low value = danger
}

function Gauge({ label, value, rawValue, threshold, inverted = false }: GaugeProps) {
  const danger = inverted ? value < threshold : value > threshold;
  const fill = danger ? "#FF3366" : value > threshold * 0.7 ? "#FFD700" : "#00FFA3";
  const data = [{ value: Math.max(0, Math.min(100, value)), fill }];

  return (
    <div className="flex flex-col items-center gap-1">
      <span className="text-[10px] text-dim tracking-widest">{label}</span>
      <div className="relative w-32 h-32">
        <ResponsiveContainer width="100%" height="100%">
          <RadialBarChart
            cx="50%"
            cy="50%"
            innerRadius="60%"
            outerRadius="90%"
            startAngle={220}
            endAngle={-40}
            data={data}
          >
            <PolarAngleAxis type="number" domain={[0, 100]} tick={false} />
            <RadialBar
              background={{ fill: "#2A2A2A" }}
              dataKey="value"
              cornerRadius={4}
            />
          </RadialBarChart>
        </ResponsiveContainer>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className={`text-lg font-bold ${danger ? "text-red" : "text-green"}`}>
            {rawValue}
          </span>
        </div>
      </div>
      <div className={`text-[9px] tracking-wider ${danger ? "text-red" : "text-dim"}`}>
        {danger ? "▲ DANGER" : "● NORMAL"}
      </div>
    </div>
  );
}

interface Props {
  garchPersistence?: number;
  capePercentile?: number;
  yieldSpreadBps?: number;
}

export function CrisisThermometer({ garchPersistence, capePercentile, yieldSpreadBps }: Props) {
  const garchNorm = garchPersistence != null ? garchPersistence * 100 : 0;
  const spreadNorm = yieldSpreadBps != null
    ? Math.max(0, Math.min(100, 50 + yieldSpreadBps / 4))
    : 50;

  return (
    <div className="bg-surface border border-border p-4">
      <div className="text-[10px] text-dim tracking-widest mb-4">CRISIS THERMOMETER</div>
      <div className="flex justify-around gap-4">
        <Gauge
          label="GARCH PERSISTENCE"
          value={garchNorm}
          rawValue={garchPersistence != null ? garchPersistence.toFixed(3) : "—"}
          threshold={98}
        />
        <Gauge
          label="CAPE PERCENTILE"
          value={capePercentile ?? 0}
          rawValue={capePercentile != null ? `P${capePercentile.toFixed(0)}` : "—"}
          threshold={95}
        />
        <Gauge
          label="YIELD SPREAD"
          value={spreadNorm}
          rawValue={yieldSpreadBps != null ? `${yieldSpreadBps.toFixed(0)}bps` : "—"}
          threshold={50}
          inverted
        />
      </div>
    </div>
  );
}
