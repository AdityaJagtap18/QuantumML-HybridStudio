export function TopBar() {
  return (
    <header className="flex h-11 shrink-0 items-center justify-between border-b border-border bg-surface px-4">
      <div className="flex items-center gap-2">
        <span className="flex h-5 w-5 items-center justify-center bg-accent text-[10px] font-bold text-white">
          H
        </span>
        <span className="text-sm font-semibold tracking-tight text-ink">Hybrid Studio</span>
        <span className="hidden text-xs text-muted sm:inline">
          Hybrid Quantum-Classical Neural Network Research Environment
        </span>
      </div>
      <div className="flex items-center gap-3 text-xs text-muted">
        <span className="font-mono-tech">PyTorch + Qiskit ML</span>
      </div>
    </header>
  );
}
