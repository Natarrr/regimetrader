// app/page.tsx — Crisis Thermometer main dashboard (server component)
import { Suspense } from "react";
import { api } from "@/lib/api";
import { MinskyAlert } from "@/components/MinskyAlert";
import { CrisisThermometer } from "@/components/CrisisThermometer";
import { RegimeCard } from "@/components/RegimeCard";

async function DashboardData() {
  // Parallel fetch all panel data
  const [minsky, monetary, garch, cape, regime] = await Promise.allSettled([
    api.minsky(),
    api.monetaryPulse(),
    api.garch("SPY"),
    api.cape(),
    api.regime("SPY"),
  ]);

  const minskyData = minsky.status === "fulfilled" ? minsky.value : null;
  const monetaryData = monetary.status === "fulfilled" ? monetary.value : null;
  const garchData = garch.status === "fulfilled" ? garch.value : null;
  const capeData = cape.status === "fulfilled" ? cape.value : null;
  const regimeData = regime.status === "fulfilled" ? regime.value : null;

  return (
    <div className="flex flex-col gap-3">
      {/* Minsky banner */}
      <MinskyAlert data={minskyData} />

      {/* Crisis thermometer gauges */}
      <CrisisThermometer
        garchPersistence={garchData?.persistence}
        capePercentile={capeData?.cape_percentile}
        yieldSpreadBps={monetaryData?.yield_spread_bps}
      />

      {/* Regime card */}
      {regimeData && (
        <RegimeCard
          symbol={regimeData.symbol}
          regime={regimeData.laureate_regime}
          hmm_label={regimeData.hmm_label}
          monetary_regime={regimeData.monetary_regime}
          volatility_regime={regimeData.volatility_regime}
          position_scale={regimeData.position_scale}
          is_uncertain={regimeData.is_uncertain}
        />
      )}

      {/* Metrics grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <MetricCard
          label="SHILLER CAPE"
          value={capeData?.cape.toFixed(1) ?? "—"}
          sub={capeData ? `P${capeData.cape_percentile.toFixed(0)} vs 40yr` : ""}
          danger={capeData?.is_danger_zone}
        />
        <MetricCard
          label="GARCH PERSIST."
          value={garchData?.persistence.toFixed(4) ?? "—"}
          sub={garchData?.volatility_regime ?? ""}
          danger={garchData ? garchData.persistence > 0.98 : false}
        />
        <MetricCard
          label="YIELD SPREAD"
          value={monetaryData ? `${monetaryData.yield_spread_bps.toFixed(0)} bps` : "—"}
          sub={monetaryData?.monetary_regime ?? ""}
          danger={monetaryData?.is_inverted}
        />
        <MetricCard
          label="EXCESS CAPE YLD"
          value={capeData ? `${(capeData.ecy * 100).toFixed(2)}%` : "—"}
          sub={capeData && capeData.ecy < 0 ? "EQUITIES EXPENSIVE" : "NORMAL"}
          danger={capeData ? capeData.ecy < 0 : false}
        />
      </div>

      {/* Contagion link */}
      <div className="bg-surface border border-border p-3 text-xs">
        <a href="/contagion" className="text-blue hover:text-green underline tracking-wider">
          → LEONTIEF CONTAGION WEB — simulate sector shock propagation
        </a>
      </div>
    </div>
  );
}

function MetricCard({
  label,
  value,
  sub,
  danger,
}: {
  label: string;
  value: string;
  sub?: string;
  danger?: boolean;
}) {
  return (
    <div className="bg-surface border border-border p-3">
      <div className="text-[9px] text-dim tracking-widest mb-1">{label}</div>
      <div className={`text-xl font-bold ${danger ? "text-red" : "text-text"}`}>{value}</div>
      {sub && <div className="text-[9px] text-dim mt-1">{sub}</div>}
    </div>
  );
}

export default function Home() {
  return (
    <main className="min-h-screen bg-bg p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4 border-b border-border pb-2">
        <div>
          <h1 className="text-sm font-bold tracking-widest text-green">THE LAUREATE ENGINE</h1>
          <p className="text-[9px] text-dim">Nobel Prize-powered market regime detection</p>
        </div>
        <nav className="flex gap-4 text-[10px] text-dim">
          <a href="/" className="text-green">DASHBOARD</a>
          <a href="/contagion" className="hover:text-text">CONTAGION</a>
        </nav>
      </div>

      {/* Main content */}
      <Suspense
        fallback={
          <div className="text-dim text-xs animate-pulse">CONNECTING TO LAUREATE ENGINE...</div>
        }
      >
        <DashboardData />
      </Suspense>
    </main>
  );
}
