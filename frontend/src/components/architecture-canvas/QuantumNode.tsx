import { Atom } from "lucide-react";

export function QuantumNode({
  nQubits,
  selected,
  onSelect,
}: {
  nQubits: number;
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <div className="flex flex-col items-center gap-1.5">
      <button
        onClick={onSelect}
        className={`flex min-h-[7rem] w-24 flex-col items-center justify-center gap-1 border-2 px-2 py-3 transition-colors ${
          selected ? "border-quantum bg-quantum/10" : "border-quantum/50 bg-surface hover:bg-quantum/5"
        }`}
      >
        <Atom size={22} className="text-quantum" />
        <span className="font-mono-tech text-[10px] text-quantum">{nQubits} qubits</span>
      </button>
      <span className="text-[10px] font-medium uppercase tracking-wide text-quantum">Quantum Layer</span>
      <span className="font-mono-tech text-[10px] text-muted">TorchConnector</span>
    </div>
  );
}
