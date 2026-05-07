"use client";

import { useEffect, useRef } from "react";
import * as d3 from "d3";
import type { ContagionOut } from "@/lib/types";

interface Props {
  data: ContagionOut | null;
}

const SECTORS = [
  "Energy","Materials","Industrials","Consumer_Disc","Consumer_Stap",
  "Healthcare","Financials","IT","Comm_Services","Utilities","Real_Estate",
];

export function LeontiefWeb({ data }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !data) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const W = svgRef.current.clientWidth || 500;
    const H = 400;

    const nodes = SECTORS.map((id) => ({
      id,
      impact: data.sector_impacts[id] ?? 0,
    }));

    // Simple links: every sector connects to the shocked sector
    const links = SECTORS.filter((s) => s !== data.shock_sector).map((s) => ({
      source: data.shock_sector,
      target: s,
    }));

    const color = d3.scaleLinear<string>()
      .domain([-20, -5, 0])
      .range(["#FF3366", "#FFD700", "#00FFA3"]);

    const sim = d3.forceSimulation(nodes as d3.SimulationNodeDatum[])
      .force("link", d3.forceLink(links).id((d: d3.SimulationNodeDatum) => (d as {id: string}).id).distance(120))
      .force("charge", d3.forceManyBody().strength(-200))
      .force("center", d3.forceCenter(W / 2, H / 2));

    const link = svg.append("g")
      .selectAll("line")
      .data(links)
      .join("line")
      .attr("stroke", "#2A2A2A")
      .attr("stroke-width", 1);

    const node = svg.append("g")
      .selectAll("g")
      .data(nodes)
      .join("g")
      .attr("cursor", "pointer");

    node.append("circle")
      .attr("r", 18)
      .attr("fill", (d) => color(d.impact))
      .attr("stroke", "#050505")
      .attr("stroke-width", 2);

    node.append("text")
      .attr("text-anchor", "middle")
      .attr("dy", "0.35em")
      .attr("fill", "#050505")
      .attr("font-size", "7px")
      .attr("font-family", "monospace")
      .text((d) => d.id.replace("_", "\n").slice(0, 6));

    node.append("title")
      .text((d) => `${d.id}: ${d.impact.toFixed(2)}%`);

    sim.on("tick", () => {
      link
        .attr("x1", (d: d3.SimulationLinkDatum<d3.SimulationNodeDatum>) => (d.source as d3.SimulationNodeDatum & {x?: number}).x ?? 0)
        .attr("y1", (d: d3.SimulationLinkDatum<d3.SimulationNodeDatum>) => (d.source as d3.SimulationNodeDatum & {y?: number}).y ?? 0)
        .attr("x2", (d: d3.SimulationLinkDatum<d3.SimulationNodeDatum>) => (d.target as d3.SimulationNodeDatum & {x?: number}).x ?? 0)
        .attr("y2", (d: d3.SimulationLinkDatum<d3.SimulationNodeDatum>) => (d.target as d3.SimulationNodeDatum & {y?: number}).y ?? 0);

      node.attr("transform", (d: d3.SimulationNodeDatum) =>
        `translate(${(d as d3.SimulationNodeDatum & {x?: number}).x ?? 0},${(d as d3.SimulationNodeDatum & {y?: number}).y ?? 0})`
      );
    });

    return () => { sim.stop(); };
  }, [data]);

  return (
    <div className="bg-surface border border-border p-4">
      <div className="text-[10px] text-dim tracking-widest mb-2">LEONTIEF CONTAGION WEB</div>
      {!data ? (
        <div className="text-dim text-xs">Select a sector shock to visualise propagation.</div>
      ) : (
        <>
          <div className="text-xs text-dim mb-2">
            Shock: <span className="text-red font-bold">{data.shock_sector} −20%</span>
            {" | "}GDP Impact: <span className="text-red">{data.total_gdp_impact_pct.toFixed(2)}%</span>
            {" | "}Critical nodes: <span className="text-yellow">{data.critical_nodes.join(", ")}</span>
          </div>
          <svg ref={svgRef} width="100%" height={400} className="bg-bg rounded" />
        </>
      )}
    </div>
  );
}
