// Statyczne diagramy SVG pokazujące, jak każdy z modeli klasyfikuje.

const G = "var(--green)";
const R = "var(--red)";
const A = "var(--accent)";
const L = "var(--line-strong)";
const INK = "var(--ink)";
const MUT = "var(--muted)";

const svgProps = {
  viewBox: "0 0 300 210",
  width: "100%",
  style: { maxWidth: 440, height: "auto" as const },
};

// ---------------- KNN ----------------
export function VizKNN() {
  const greens = [
    [28, 55], [62, 168], [98, 26], [262, 42], [278, 120], [248, 178], [38, 112], [206, 28],
  ];
  const reds = [
    [135, 78], [176, 74], [120, 122], [186, 122], [150, 152], [150, 52],
  ];
  const qx = 150, qy = 102;
  return (
    <svg {...svgProps} role="img" aria-label="Diagram KNN">
      <circle cx={qx} cy={qy} r={60} fill="none" stroke={A} strokeWidth={1.5} strokeDasharray="4 4" opacity={0.6} />
      {reds.map(([x, y], i) => (
        <line key={"l" + i} x1={qx} y1={qy} x2={x} y2={y} stroke={A} strokeWidth={1} opacity={0.35} />
      ))}
      {greens.map(([x, y], i) => (
        <circle key={"g" + i} cx={x} cy={y} r={5} fill={G} />
      ))}
      {reds.map(([x, y], i) => (
        <circle key={"r" + i} cx={x} cy={y} r={5} fill={R} />
      ))}
      <circle cx={qx} cy={qy} r={10} fill={INK} />
      <text x={qx} y={qy + 3.5} textAnchor="middle" fontSize={11} fontWeight={700} fill="#fff">?</text>
      <text x={qx} y={200} textAnchor="middle" fontSize={10} fill={MUT}>
        większość sąsiadów = bankrut
      </text>
    </svg>
  );
}

// ---------------- Drzewo decyzyjne ----------------
function Box({ x, y, w, h, t, fill, stroke, color, fs = 9.5 }: any) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={7} fill={fill} stroke={stroke} strokeWidth={1.2} />
      <text x={x + w / 2} y={y + h / 2 + 3.5} textAnchor="middle" fontSize={fs} fontWeight={600} fill={color}>
        {t}
      </text>
    </g>
  );
}
function Edge({ x1, y1, x2, y2, label, lx, ly }: any) {
  return (
    <g>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={L} strokeWidth={1.5} />
      {label && (
        <text x={lx} y={ly} textAnchor="middle" fontSize={8.5} fill={MUT} fontWeight={600}>
          {label}
        </text>
      )}
    </g>
  );
}
export function VizTree() {
  return (
    <svg {...svgProps} role="img" aria-label="Diagram drzewa decyzyjnego">
      <Edge x1={130} y1={44} x2={70} y2={86} label="nie" lx={86} ly={70} />
      <Edge x1={170} y1={44} x2={230} y2={86} label="tak" lx={214} ly={70} />
      <Edge x1={70} y1={120} x2={35} y2={158} label="nie" lx={40} ly={144} />
      <Edge x1={70} y1={120} x2={120} y2={158} label="tak" lx={108} ly={144} />
      <Box x={90} y={12} w={120} h={32} t="Zadłużenie > 0,8?" fill="var(--accent-soft)" stroke={A} color={A} />
      <Box x={18} y={88} w={104} h={32} t="Rentowność < 0?" fill="var(--accent-soft)" stroke={A} color={A} />
      <Box x={196} y={89} w={88} h={30} t="Bankrut" fill={R} stroke={R} color="#fff" />
      <Box x={6} y={160} w={84} h={30} t="Zdrowa" fill={G} stroke={G} color="#fff" />
      <Box x={100} y={160} w={84} h={30} t="Bankrut" fill={R} stroke={R} color="#fff" />
    </svg>
  );
}

// ---------------- Las losowy ----------------
function MiniTree({ cx, cy, vote }: { cx: number; cy: number; vote: "g" | "r" }) {
  const leaf = vote === "g" ? G : R;
  return (
    <g>
      <line x1={cx} y1={cy} x2={cx - 16} y2={cy + 26} stroke={L} strokeWidth={1.6} />
      <line x1={cx} y1={cy} x2={cx + 16} y2={cy + 26} stroke={L} strokeWidth={1.6} />
      <circle cx={cx} cy={cy} r={6} fill={A} />
      <circle cx={cx - 16} cy={cy + 26} r={5} fill={vote === "g" ? G : "var(--line-strong)"} />
      <circle cx={cx + 16} cy={cy + 26} r={5} fill={leaf} />
    </g>
  );
}
export function VizForest() {
  const xs = [55, 150, 245];
  const votes: ("g" | "r")[] = ["r", "g", "r"];
  return (
    <svg {...svgProps} role="img" aria-label="Diagram lasu losowego">
      {xs.map((x, i) => (
        <g key={i}>
          <MiniTree cx={x} cy={26} vote={votes[i]} />
          <rect x={x - 30} y={74} width={60} height={22} rx={11} fill={votes[i] === "g" ? G : R} />
          <text x={x} y={89} textAnchor="middle" fontSize={9.5} fontWeight={700} fill="#fff">
            {votes[i] === "g" ? "zdrowa" : "bankrut"}
          </text>
          <line x1={x} y1={98} x2={150} y2={150} stroke={L} strokeWidth={1.4} />
        </g>
      ))}
      <rect x={92} y={150} width={116} height={36} rx={9} fill={R} />
      <text x={150} y={167} textAnchor="middle" fontSize={10} fontWeight={700} fill="#fff">BANKRUT</text>
      <text x={150} y={179} textAnchor="middle" fontSize={8} fill="#fff" opacity={0.9}>2 z 3 głosów</text>
      <text x={150} y={204} textAnchor="middle" fontSize={10} fill={MUT}>300 drzew głosuje większością</text>
    </svg>
  );
}

// ---------------- Gradient Boosting ----------------
export function VizBoosting() {
  const xs = [48, 150, 252];
  const errH = [46, 26, 12]; // malejący błąd
  return (
    <svg {...svgProps} role="img" aria-label="Diagram gradient boosting">
      {xs.map((x, i) => (
        <g key={i}>
          <line x1={x} y1={20} x2={x - 14} y2={42} stroke={L} strokeWidth={1.6} />
          <line x1={x} y1={20} x2={x + 14} y2={42} stroke={L} strokeWidth={1.6} />
          <circle cx={x} cy={20} r={6} fill={A} />
          <circle cx={x - 14} cy={42} r={4.5} fill={G} />
          <circle cx={x + 14} cy={42} r={4.5} fill={R} />
          <text x={x} y={66} textAnchor="middle" fontSize={8.5} fill={MUT}>{`drzewo ${i + 1}`}</text>
          {/* słupek błędu */}
          <rect x={x - 9} y={150 - errH[i]} width={18} height={errH[i]} rx={3} fill={R} opacity={0.85} />
          <line x1={x - 14} y1={150} x2={x + 14} y2={150} stroke={L} strokeWidth={1.2} />
        </g>
      ))}
      {/* strzałki "popraw błąd" */}
      <text x={99} y={36} textAnchor="middle" fontSize={13} fill={A}>→</text>
      <text x={201} y={36} textAnchor="middle" fontSize={13} fill={A}>→</text>
      <text x={99} y={26} textAnchor="middle" fontSize={7.5} fill={MUT}>popraw</text>
      <text x={201} y={26} textAnchor="middle" fontSize={7.5} fill={MUT}>popraw</text>
      <text x={150} y={172} textAnchor="middle" fontSize={9} fill={MUT}>błąd maleje z każdym drzewem</text>
      <text x={150} y={196} textAnchor="middle" fontSize={10} fontWeight={600} fill={INK}>
        każde drzewo naprawia poprzednie
      </text>
    </svg>
  );
}
