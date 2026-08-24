import React from "react";

/**
 * The signature visual element of the dashboard: an instrument-dial style
 * horizontal gauge (evoking an oscilloscope / seismograph readout) rather
 * than a generic circular progress ring. Hash marks read like a lab
 * instrument; the needle position communicates AI probability continuously,
 * resisting the pull toward a binary real/fake readout.
 */
export default function ProbabilityGauge({ probability, confidence }) {
  const pct = Math.round(probability * 100);
  const needleX = 12 + probability * 376; // track spans x=12..388 in the 400-wide viewbox

  return (
    <div className="w-full">
      <svg viewBox="0 0 400 96" className="w-full h-24" role="img" aria-label={`AI probability ${pct} percent`}>
        <defs>
          <linearGradient id="gaugeGradient" x1="0" x2="1" y1="0" y2="0">
            <stop offset="0%" stopColor="#4CC9C0" />
            <stop offset="50%" stopColor="#E8A33D" />
            <stop offset="100%" stopColor="#E2574C" />
          </linearGradient>
        </defs>

        {/* track */}
        <rect x="12" y="46" width="376" height="6" rx="3" fill="url(#gaugeGradient)" opacity="0.85" />

        {/* hash marks, every 10% */}
        {Array.from({ length: 11 }).map((_, i) => {
          const x = 12 + (i / 10) * 376;
          const major = i % 5 === 0;
          return (
            <line
              key={i}
              x1={x}
              y1={major ? 30 : 36}
              x2={x}
              y2={46}
              stroke="#7C8896"
              strokeWidth={major ? 1.5 : 1}
              opacity={major ? 0.8 : 0.4}
            />
          );
        })}

        {/* confidence band around the needle, width shrinks as confidence rises */}
        <rect
          x={needleX - (1 - confidence) * 40 - 4}
          y="40"
          width={(1 - confidence) * 80 + 8}
          height="18"
          rx="4"
          fill="#E9EDF1"
          opacity="0.08"
        />

        {/* needle */}
        <line x1={needleX} y1="20" x2={needleX} y2="66" stroke="#E9EDF1" strokeWidth="2" />
        <polygon points={`${needleX - 6},20 ${needleX + 6},20 ${needleX},10`} fill="#E9EDF1" />

        {/* labels */}
        <text x="12" y="90" fontSize="11" fill="#7C8896" fontFamily="'IBM Plex Mono', monospace">
          0% — authentic
        </text>
        <text x="388" y="90" fontSize="11" fill="#7C8896" fontFamily="'IBM Plex Mono', monospace" textAnchor="end">
          100% — AI-generated
        </text>
      </svg>
    </div>
  );
}
