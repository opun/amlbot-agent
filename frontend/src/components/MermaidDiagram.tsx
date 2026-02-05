"use client";

import { useEffect, useRef, useState } from "react";
import mermaid from "mermaid";

interface MermaidDiagramProps {
  chart: string;
}

// Initialize mermaid once
let mermaidInitialized = false;

function initializeMermaid() {
  if (mermaidInitialized) return;

  mermaid.initialize({
    startOnLoad: false,
    theme: "dark",
    themeVariables: {
      primaryColor: "#10b981",
      primaryTextColor: "#ffffff",
      primaryBorderColor: "#059669",
      lineColor: "#6b7280",
      secondaryColor: "#1f2937",
      tertiaryColor: "#111827",
      background: "#1a1a1a",
      mainBkg: "#1a1a1a",
      secondBkg: "#111827",
      textColor: "#ffffff",
      textColor2: "#ffffff", // Improved contrast for secondary text
      edgeLabelBackground: "#1f2937",
      edgeLabelColor: "#ffffff", // White text on edge labels for readability
      clusterBkg: "#111827",
      clusterBorder: "#374151",
      defaultLinkColor: "#10b981",
      titleColor: "#ffffff",
      actorBorder: "#6b7280",
      actorBkg: "#1f2937",
      actorTextColor: "#ffffff",
      actorLineColor: "#6b7280",
      signalColor: "#ffffff",
      signalTextColor: "#ffffff",
      labelBoxBkgColor: "#1f2937",
      labelBoxBorderColor: "#6b7280",
      labelTextColor: "#ffffff",
      loopTextColor: "#ffffff",
      noteBorderColor: "#6b7280",
      noteBkgColor: "#111827",
      noteTextColor: "#ffffff",
      activationBorderColor: "#10b981",
      activationBkgColor: "#065f46",
      sequenceNumberColor: "#ffffff",
      sectionBkgColor: "#111827",
      altBkgColor: "#1f2937",
      elseBkgColor: "#111827",
      // Additional variables for better node text readability
      cScale0: "#10b981",
      cScale1: "#059669",
      cScale2: "#047857",
    },
  });

  mermaidInitialized = true;
}

export function MermaidDiagram({ chart }: MermaidDiagramProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!containerRef.current || !chart) {
      setError(null);
      return;
    }

    // Initialize mermaid
    initializeMermaid();

    // Clear previous content and error
    containerRef.current.innerHTML = "";
    setError(null);

    // Create a unique ID for this diagram
    const id = `mermaid-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

    // Render the diagram
    mermaid
      .render(id, chart)
      .then((result) => {
        if (containerRef.current) {
          containerRef.current.innerHTML = result.svg;

          // Apply custom styles for better readability
          const svg = containerRef.current.querySelector("svg");
          if (svg) {
            // Add style element to ensure text readability
            const style = document.createElement("style");
            style.textContent = `
              .mermaid svg .node.unknown text,
              .mermaid svg .node.unknown .nodeLabel,
              .mermaid svg .node.unknown tspan {
                fill: #111827 !important;
                font-weight: 600 !important;
                font-size: 14px !important;
              }
              .mermaid svg .node.victim text,
              .mermaid svg .node.victim .nodeLabel,
              .mermaid svg .node.victim tspan,
              .mermaid svg .node.perpetrator text,
              .mermaid svg .node.perpetrator .nodeLabel,
              .mermaid svg .node.perpetrator tspan {
                fill: #000000 !important;
                font-weight: 700 !important;
                font-size: 14px !important;
              }
              .mermaid svg .node.service text,
              .mermaid svg .node.service .nodeLabel,
              .mermaid svg .node.service tspan {
                fill: #000000 !important;
                font-weight: 600 !important;
                font-size: 14px !important;
              }
              .mermaid svg .edgeLabel {
                fill: #ffffff !important;
                background-color: #1f2937 !important;
              }
              .mermaid svg .edgeLabel text,
              .mermaid svg .edgeLabel tspan {
                fill: #ffffff !important;
                font-weight: 500 !important;
                font-size: 12px !important;
              }
              .mermaid svg .node rect {
                rx: 6px;
                ry: 6px;
              }
            `;
            svg.appendChild(style);
          }
        }
      })
      .catch((err) => {
        console.error("Mermaid rendering error:", err);
        setError(err instanceof Error ? err.message : "Failed to render diagram");
        if (containerRef.current) {
          containerRef.current.innerHTML = "";
        }
      });
  }, [chart]);

  if (!chart) {
    return null;
  }

  return (
    <div className="bg-gray-900 rounded-lg p-4 overflow-x-auto" style={{ minHeight: "200px" }}>
      {error ? (
        <div className="text-red-400 text-sm p-4">Error rendering diagram: {error}</div>
      ) : (
        <div ref={containerRef} className="mermaid" />
      )}
    </div>
  );
}
