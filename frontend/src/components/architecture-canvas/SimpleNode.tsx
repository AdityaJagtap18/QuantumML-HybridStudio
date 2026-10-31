export function SimpleNode({
  label,
  sublabel,
  selected,
  onSelect,
}: {
  label: string;
  sublabel?: string;
  selected?: boolean;
  onSelect?: () => void;
}) {
  const Tag = onSelect ? "button" : "div";
  return (
    <div className="flex flex-col items-center gap-1.5">
      <Tag
        onClick={onSelect}
        className={`flex h-20 w-20 flex-col items-center justify-center border px-2 text-center transition-colors ${
          selected ? "border-accent bg-accent/10" : "border-border bg-surface"
        } ${onSelect ? "hover:bg-canvas" : ""}`}
      >
        <span className="font-mono-tech text-[10px] text-ink">{sublabel}</span>
      </Tag>
      <span className="text-[10px] font-medium uppercase tracking-wide text-ink">{label}</span>
    </div>
  );
}
