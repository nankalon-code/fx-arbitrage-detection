import { useEffect, useRef, useCallback } from "react";
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

export function NetworkGraph({ opportunities }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const simRef = useRef<d3.Simulation<Node, Link> | null>(null);
  const initializedRef = useRef(false);
  const nodesRef = useRef<Node[]>([]);
  const linksRef = useRef<Link[]>([]);

  // Build the graph once, then update arb highlights in-place
  const initGraph = useCallback(() => {
    if (!svgRef.current || initializedRef.current) return;
    initializedRef.current = true;

    const svg = d3.select(svgRef.current);
    const width = svgRef.current.clientWidth || 600;
    const height = svgRef.current.clientHeight || 300;

    svg.selectAll("*").remove();

    const nodes: Node[] = CURRENCIES.map(id => ({ id, isArb: false }));
    nodesRef.current = nodes;

    const links: Link[] = [];
    CURRENCIES.forEach(src => {
      Object.entries(BASE_RATES[src] || {}).forEach(([tgt, rate]) => {
        links.push({ source: src, target: tgt, rate, isArb: false });
      });
    });
    linksRef.current = links;

    const defs = svg.append("defs");

    defs.append("marker").attr("id", "arrow-normal")
      .attr("viewBox", "0 -4 8 8").attr("refX", 22).attr("refY", 0)
      .attr("markerWidth", 5).attr("markerHeight", 5).attr("orient", "auto")
      .append("path").attr("d", "M0,-4L8,0L0,4").attr("fill", "#1a2a4a");

    defs.append("marker").attr("id", "arrow-arb")
      .attr("viewBox", "0 -4 8 8").attr("refX", 22).attr("refY", 0)
      .attr("markerWidth", 5).attr("markerHeight", 5).attr("orient", "auto")
      .append("path").attr("d", "M0,-4L8,0L0,4").attr("fill", "#00ff88");

    const filter = defs.append("filter").attr("id", "glow");
    filter.append("feGaussianBlur").attr("stdDeviation", "3").attr("result", "blur");
    const merge = filter.append("feMerge");
    merge.append("feMergeNode").attr("in", "blur");
    merge.append("feMergeNode").attr("in", "SourceGraphic");

    const g = svg.append("g").attr("class", "graph-container");

    g.append("g").attr("class", "links-group").selectAll("line")
      .data(links).join("line")
      .attr("class", "graph-link")
      .attr("stroke", "#1a2a4a").attr("stroke-width", 0.8)
      .attr("stroke-opacity", 0.5)
      .attr("marker-end", "url(#arrow-normal)");

    const nodeG = g.append("g").attr("class", "nodes-group").selectAll("g")
      .data(nodes).join("g").attr("cursor", "pointer")
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

    nodeG.append("circle").attr("r", 16)
      .attr("fill", "rgba(10,22,40,0.9)").attr("stroke", "#1a2a4a").attr("stroke-width", 1);

    nodeG.append("text").text((d: Node) => d.id)
      .attr("text-anchor", "middle").attr("dominant-baseline", "central")
      .attr("font-family", "JetBrains Mono, monospace")
      .attr("font-size", "9px").attr("font-weight", "500").attr("fill", "#94a3b8")
      .attr("pointer-events", "none");

    // Tooltip
    const tooltip = d3.select("body").select<HTMLDivElement>(".d3-tooltip");
    nodeG.on("mouseover", function (event: MouseEvent, d: Node) {
      const rates = BASE_RATES[d.id] || {};
      const rateText = Object.entries(rates).slice(0, 5).map(([k, v]) => `${d.id}→${k}: ${v}`).join("\n");
      tooltip.style("display", "block")
        .style("left", (event.pageX + 12) + "px").style("top", (event.pageY - 12) + "px")
        .html(`<strong style="color:var(--accent-blue)">${d.id}</strong><br/><pre style="margin:4px 0 0;font-size:10px;color:var(--text-muted)">${rateText}</pre>`);
    }).on("mousemove", function (event: MouseEvent) {
      tooltip.style("left", (event.pageX + 12) + "px").style("top", (event.pageY - 12) + "px");
    }).on("mouseout", () => { tooltip.style("display", "none"); });

    // Profit label
    svg.append("text").attr("class", "arb-label")
      .attr("x", width / 2).attr("y", 24).attr("text-anchor", "middle")
      .attr("font-family", "JetBrains Mono, monospace").attr("font-size", "13px")
      .attr("font-weight", "700").attr("fill", "#00ff88").text("");

    const sim = d3.forceSimulation<Node>(nodes)
      .force("link", d3.forceLink<Node, Link>(links).id((d: Node) => d.id).distance(90).strength(0.4))
      .force("charge", d3.forceManyBody().strength(-300))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide(30))
      .on("tick", () => {
        g.select(".links-group").selectAll<SVGLineElement, Link>("line")
          .attr("x1", (d: any) => d.source.x).attr("y1", (d: any) => d.source.y)
          .attr("x2", (d: any) => d.target.x).attr("y2", (d: any) => d.target.y);
        g.select(".nodes-group").selectAll<SVGGElement, Node>("g")
          .attr("transform", (d: any) => `translate(${d.x},${d.y})`);
      });

    simRef.current = sim;
  }, []);

  // Initialize graph on mount
  useEffect(() => {
    initGraph();
    return () => {
      if (simRef.current) simRef.current.stop();
      initializedRef.current = false;
    };
  }, [initGraph]);

  // Update highlights WITHOUT recreating the simulation
  useEffect(() => {
    if (!svgRef.current || !initializedRef.current) return;
    const svg = d3.select(svgRef.current);

    const arbCurrencies = new Set<string>();
    const arbEdgeSet = new Set<string>();
    let labelText = "";

    if (opportunities.length > 0) {
      const best = opportunities[0];
      if (Array.isArray(best.path)) {
        best.path.forEach(c => arbCurrencies.add(c));
        for (let i = 0; i < best.path.length - 1; i++) {
          arbEdgeSet.add(`${best.path[i]}-${best.path[i + 1]}`);
        }
        labelText = `⚡ Best: +${best.profit_bps.toFixed(1)} bps  ·  ${best.path.join(" → ")}`;
      }
    }

    // Update link colors
    svg.select(".links-group").selectAll<SVGLineElement, Link>("line")
      .attr("stroke", (d: any) => {
        const key = `${typeof d.source === 'object' ? d.source.id : d.source}-${typeof d.target === 'object' ? d.target.id : d.target}`;
        return arbEdgeSet.has(key) ? "#00ff88" : "#1a2a4a";
      })
      .attr("stroke-width", (d: any) => {
        const key = `${typeof d.source === 'object' ? d.source.id : d.source}-${typeof d.target === 'object' ? d.target.id : d.target}`;
        return arbEdgeSet.has(key) ? 2.5 : 0.8;
      })
      .attr("stroke-opacity", (d: any) => {
        const key = `${typeof d.source === 'object' ? d.source.id : d.source}-${typeof d.target === 'object' ? d.target.id : d.target}`;
        return arbEdgeSet.has(key) ? 1 : 0.5;
      })
      .attr("marker-end", (d: any) => {
        const key = `${typeof d.source === 'object' ? d.source.id : d.source}-${typeof d.target === 'object' ? d.target.id : d.target}`;
        return arbEdgeSet.has(key) ? "url(#arrow-arb)" : "url(#arrow-normal)";
      })
      .attr("filter", (d: any) => {
        const key = `${typeof d.source === 'object' ? d.source.id : d.source}-${typeof d.target === 'object' ? d.target.id : d.target}`;
        return arbEdgeSet.has(key) ? "url(#glow)" : null;
      });

    // Update node colors
    svg.select(".nodes-group").selectAll<SVGGElement, Node>("g").each(function (d: Node) {
      const isArb = arbCurrencies.has(d.id);
      d3.select(this).select("circle")
        .attr("r", isArb ? 22 : 16)
        .attr("fill", isArb ? "rgba(0,255,136,0.15)" : "rgba(10,22,40,0.9)")
        .attr("stroke", isArb ? "#00ff88" : "#1a2a4a")
        .attr("stroke-width", isArb ? 2 : 1)
        .attr("filter", isArb ? "url(#glow)" : null);
      d3.select(this).select("text")
        .attr("font-size", isArb ? "11px" : "9px")
        .attr("font-weight", isArb ? "700" : "500")
        .attr("fill", isArb ? "#00ff88" : "#94a3b8");
    });

    // Update label
    svg.select(".arb-label").text(labelText);
  }, [opportunities]);

  return (
    <div style={{ position: "relative", width: "100%", height: 300 }}>
      <svg ref={svgRef} style={{ width: "100%", height: "100%", overflow: "visible" }} />
    </div>
  );
}
