import { useEffect, useRef } from "react";
import * as d3 from "d3";
import type { Opportunity } from "../types";

interface Props {
  opportunities: Opportunity[];
  prices: Record<string, { bid: number; ask: number; mid: number; vol: number }>;
}

const CURRENCIES = ["EUR", "USD", "JPY", "GBP", "CHF", "AUD", "CAD", "NZD"];

const BASE_RATES: Record<string, Record<string, number>> = {
  EUR: { USD: 1.085, GBP: 0.854, JPY: 162.2, CHF: 0.965, AUD: 1.656, CAD: 1.475, NZD: 1.78 },
  USD: { EUR: 0.922, GBP: 0.787, JPY: 149.5, CHF: 0.890, AUD: 1.527, CAD: 1.360, NZD: 1.653 },
  GBP: { EUR: 1.170, USD: 1.270, JPY: 189.8, CHF: 1.134, AUD: 1.938, CAD: 1.725, NZD: 2.08 },
  JPY: { EUR: 0.0062, USD: 0.0067, GBP: 0.0053, CHF: 0.0060, AUD: 0.0102, CAD: 0.0091, NZD: 0.011 },
  CHF: { EUR: 1.036, USD: 1.123, GBP: 0.882, JPY: 167.8, AUD: 1.716, CAD: 1.528, NZD: 1.857 },
  AUD: { EUR: 0.604, USD: 0.655, GBP: 0.516, JPY: 97.90, CHF: 0.583, CAD: 0.891, NZD: 1.085 },
  CAD: { EUR: 0.678, USD: 0.735, GBP: 0.580, JPY: 109.9, CHF: 0.654, AUD: 1.123, NZD: 1.216 },
  NZD: { EUR: 0.562, USD: 0.605, GBP: 0.481, JPY: 90.49, CHF: 0.539, AUD: 0.921, CAD: 0.822 },
};

interface Node extends d3.SimulationNodeDatum {
  id: string;
  isArb: boolean;
}

interface Link extends d3.SimulationLinkDatum<Node> {
  source: string | Node;
  target: string | Node;
  rate: number;
  isArb: boolean;
}

export function NetworkGraph({ opportunities, prices }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const simRef = useRef<d3.Simulation<Node, Link> | null>(null);

  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    const width = svgRef.current.clientWidth || 600;
    const height = svgRef.current.clientHeight || 300;

    svg.selectAll("*").remove();

    // Determine arb currencies & edges
    const arbCurrencies = new Set<string>();
    const arbEdgeSet = new Set<string>();
    if (opportunities.length > 0) {
      const best = opportunities[0];
      best.path.forEach(c => arbCurrencies.add(c));
      for (let i = 0; i < best.path.length - 1; i++) {
        arbEdgeSet.add(`${best.path[i]}-${best.path[i + 1]}`);
      }
    }

    const nodes: Node[] = CURRENCIES.map(id => ({
      id,
      isArb: arbCurrencies.has(id),
    }));

    const links: Link[] = [];
    CURRENCIES.forEach(src => {
      Object.entries(BASE_RATES[src] || {}).forEach(([tgt, rate]) => {
        links.push({
          source: src,
          target: tgt,
          rate,
          isArb: arbEdgeSet.has(`${src}-${tgt}`),
        });
      });
    });

    // Defs: arrow markers
    const defs = svg.append("defs");

    defs.append("marker")
      .attr("id", "arrow-normal")
      .attr("viewBox", "0 -4 8 8")
      .attr("refX", 22).attr("refY", 0)
      .attr("markerWidth", 5).attr("markerHeight", 5)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,-4L8,0L0,4")
      .attr("fill", "#1a2a4a");

    defs.append("marker")
      .attr("id", "arrow-arb")
      .attr("viewBox", "0 -4 8 8")
      .attr("refX", 22).attr("refY", 0)
      .attr("markerWidth", 5).attr("markerHeight", 5)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,-4L8,0L0,4")
      .attr("fill", "#00ff88");

    // Glow filter
    const filter = defs.append("filter").attr("id", "glow");
    filter.append("feGaussianBlur").attr("stdDeviation", "3").attr("result", "blur");
    const merge = filter.append("feMerge");
    merge.append("feMergeNode").attr("in", "blur");
    merge.append("feMergeNode").attr("in", "SourceGraphic");

    const g = svg.append("g");

    // Links
    const link = g.append("g").selectAll("line")
      .data(links)
      .join("line")
      .attr("stroke", (d: any) => d.isArb ? "#00ff88" : "#1a2a4a")
      .attr("stroke-width", (d: any) => d.isArb ? 2.5 : 0.8)
      .attr("stroke-opacity", (d: any) => d.isArb ? 1 : 0.5)
      .attr("marker-end", (d: any) => d.isArb ? "url(#arrow-arb)" : "url(#arrow-normal)")
      .attr("filter", (d: any) => d.isArb ? "url(#glow)" : null);

    // Nodes
    const node = g.append("g").selectAll("g")
      .data(nodes)
      .join("g")
      .attr("cursor", "pointer")
      .call(
        d3.drag<SVGGElement, Node>()
          .on("start", (event, d) => {
            if (!event.active && simRef.current) simRef.current.alphaTarget(0.3).restart();
            d.fx = d.x; d.fy = d.y;
          })
          .on("drag", (event, d) => { d.fx = event.x; d.fy = event.y; })
          .on("end", (event, d) => {
            if (!event.active && simRef.current) simRef.current.alphaTarget(0);
            d.fx = null; d.fy = null;
          }) as any
      );

    node.append("circle")
      .attr("r", (d: Node) => d.isArb ? 22 : 16)
      .attr("fill", (d: Node) => d.isArb ? "rgba(0,255,136,0.15)" : "rgba(10,22,40,0.9)")
      .attr("stroke", (d: Node) => d.isArb ? "#00ff88" : "#1a2a4a")
      .attr("stroke-width", (d: Node) => d.isArb ? 2 : 1)
      .attr("filter", (d: Node) => d.isArb ? "url(#glow)" : null);

    node.append("text")
      .text((d: Node) => d.id)
      .attr("text-anchor", "middle")
      .attr("dominant-baseline", "central")
      .attr("font-family", "JetBrains Mono, monospace")
      .attr("font-size", (d: Node) => d.isArb ? "11px" : "9px")
      .attr("font-weight", (d: Node) => d.isArb ? "700" : "500")
      .attr("fill", (d: Node) => d.isArb ? "#00ff88" : "#94a3b8")
      .attr("pointer-events", "none");

    // Tooltip
    const tooltip = d3.select("body").select<HTMLDivElement>(".d3-tooltip");

    node.on("mouseover", function (event: MouseEvent, d: Node) {
      const rates = BASE_RATES[d.id] || {};
      const rateText = Object.entries(rates)
        .slice(0, 5)
        .map(([k, v]) => `${d.id}→${k}: ${v}`)
        .join("\n");
      tooltip
        .style("display", "block")
        .style("left", (event.pageX + 12) + "px")
        .style("top",  (event.pageY - 12) + "px")
        .html(`<strong style="color:var(--accent-blue)">${d.id}</strong><br/><pre style="margin:4px 0 0;font-size:10px;color:var(--text-muted)">${rateText}</pre>`);
    }).on("mousemove", function (event: MouseEvent) {
      tooltip
        .style("left", (event.pageX + 12) + "px")
        .style("top",  (event.pageY - 12) + "px");
    }).on("mouseout", () => {
      tooltip.style("display", "none");
    });

    // Simulation
    const sim = d3.forceSimulation<Node>(nodes)
      .force("link", d3.forceLink<Node, Link>(links).id((d: Node) => d.id).distance(90).strength(0.4))
      .force("charge", d3.forceManyBody().strength(-300))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide(30))
      .on("tick", () => {
        link
          .attr("x1", (d: any) => d.source.x)
          .attr("y1", (d: any) => d.source.y)
          .attr("x2", (d: any) => d.target.x)
          .attr("y2", (d: any) => d.target.y);
        node.attr("transform", (d: any) => `translate(${d.x},${d.y})`);
      });

    simRef.current = sim;

    // Profit label
    if (opportunities.length > 0) {
      const best = opportunities[0];
      const arbNodes = nodes.filter(n => arbCurrencies.has(n.id));

      const label = svg.append("text")
        .attr("x", width / 2)
        .attr("y", 24)
        .attr("text-anchor", "middle")
        .attr("font-family", "JetBrains Mono, monospace")
        .attr("font-size", "13px")
        .attr("font-weight", "700")
        .attr("fill", "#00ff88")
        .text(`⚡ Best cycle: +${best.profit_bps.toFixed(1)} bps  ·  ${best.path.join(" → ")}`);
    }

    return () => {
      sim.stop();
      tooltip.style("display", "none");
    };
  }, [opportunities]);

  return (
    <div style={{ position: "relative", width: "100%", height: 300 }}>
      <svg ref={svgRef} style={{ width: "100%", height: "100%", overflow: "visible" }} />
    </div>
  );
}
