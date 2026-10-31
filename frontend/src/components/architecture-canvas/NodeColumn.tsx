import { Minus, Plus } from "lucide-react";

export function NodeColumn({
  label,
  sublabel,
  nodeCount,
  selected,
  onSelect,
  onAdd,
  onRemove,
  color = "classical",
}: {
  label: string;
  sublabel?: string;
  nodeCount: number;
  selected: boolean;
  onSelect: () => void;
  onAdd?: () => void;
  onRemove?: () => void;
  color?: "classical" | "quantum";
}) {
  const dots = Math.min(nodeCount, 6);
  const overflow = nodeCount - dots;
  const dotColor = color === "quantum" ? "bg-quantum" : "bg-classical";

  return (
    <div className="flex flex-col items-center gap-1.5">
      {onAdd && onRemove && (
        <div className="flex items-center gap-1">
          <button onClick={onRemove} className="flex h-4 w-4 items-center justify-center border border-border text-muted hover:bg-canvas">
            <Minus size={10} />
          </button>
          <button onClick={onAdd} className="flex h-4 w-4 items-center justify-center border border-border text-muted hover:bg-canvas">
            <Plus size={10} />
          </button>
        </div>
      )}
      <button
        onClick={onSelect}
        className={`flex min-h-[7rem] w-20 flex-col items-center justify-center gap-1.5 border px-2 py-3 transition-colors ${
          selected ? "border-accent bg-accent/10" : "border-border bg-surface hover:bg-canvas"
        }`}
      >
        {Array.from({ length: dots }).map((_, i) => (
          <span key={i} className={`h-2.5 w-2.5 rounded-full ${dotColor} opacity-80`} />
        ))}
        {overflow > 0 && <span className="text-[9px] text-muted">+{overflow}</span>}
      </button>
      <span className="text-[10px] font-medium uppercase tracking-wide text-ink">{label}</span>
      {sublabel && <span className="font-mono-tech text-[10px] text-muted">{sublabel}</span>}
    </div>
  );
}
