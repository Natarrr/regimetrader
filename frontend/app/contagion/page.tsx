"use client";

import { useState } from "react";
import { api } from "@/lib/api";
import { LeontiefWeb } from "@/components/LeontiefWeb";
import type { ContagionOut } from "@/lib/types";

const SECTORS = [
  "Energy","Materials","Industrials","Consumer_Disc","Consumer_Stap",
  "Healthcare","Financials","IT","Comm_Services","Utilities","Real_Estate",
];

export default function ContagionPage() {
  const [selected, setSelected] = useState<string>("Energy");
  const [shock, setShock] = useState<number>(-20);
  const [data, setData] = useState<ContagionOut | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function simulate() {
    setLoading(true);
    setError(null);
    try {
      const result = await api.contagionShock(selected, shock);
      setData(result);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-bg p-4 font-mono">
      <div className="flex items-center justify-between mb-4 border-b border-border pb-2">
        <div>
          <h1 className="text-sm font-bold tracking-widest text-green">LEONTIEF CONTAGION WEB</h1>
          <p className="text-[9px] text-dim">Tirole (2014 Nobel) — Supply chain shock propagation</p>
        </div>
        <a href="/" className="text-[10px] text-dim hover:text-text">← DASHBOARD</a>
      </div>

      {/* Controls */}
      <div className="bg-surface border border-border p-3 mb-3 flex flex-wrap gap-4 items-end">
        <div>
          <label className="text-[9px] text-dim block mb-1">SHOCK SECTOR</label>
          <select
            value={selected}
            onChange={(e) => setSelected(e.target.value)}
            className="bg-bg border border-border text-text text-xs px-2 py-1 focus:outline-none focus:border-green"
          >
            {SECTORS.map((s) => (
              <option key={s} value={s}>{s.replace("_", " ")}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="text-[9px] text-dim block mb-1">SHOCK MAGNITUDE</label>
          <select
            value={shock}
            onChange={(e) => setShock(Number(e.target.value))}
            className="bg-bg border border-border text-text text-xs px-2 py-1 focus:outline-none focus:border-green"
          >
            {[-5, -10, -20, -30, -50].map((v) => (
              <option key={v} value={v}>{v}%</option>
            ))}
          </select>
        </div>
        <button
          onClick={simulate}
          disabled={loading}
          className="px-4 py-1 border border-green text-green text-xs tracking-wider hover:bg-green hover:text-bg transition-colors disabled:opacity-50"
        >
          {loading ? "SIMULATING..." : "RUN SIMULATION"}
        </button>
      </div>

      {error && (
        <div className="text-red text-xs border border-red p-2 mb-3">{error}</div>
      )}

      <LeontiefWeb data={data} />

      {data && (
        <div className="mt-3 bg-surface border border-border p-3">
          <div className="text-[9px] text-dim tracking-widest mb-2">SECTOR IMPACT TABLE</div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
            {Object.entries(data.sector_impacts)
              .sort((a, b) => a[1] - b[1])
              .map(([sector, impact]) => (
                <div key={sector} className="flex justify-between text-xs border border-border px-2 py-1">
                  <span className="text-dim">{sector.replace("_", " ")}</span>
                  <span className={impact < 0 ? "text-red" : "text-green"}>
                    {impact.toFixed(2)}%
                  </span>
                </div>
              ))}
          </div>
        </div>
      )}
    </main>
  );
}
