export function PlaceholderPanel({ title, description, phase }: { title: string; description: string; phase: string }) {
  return (
    <div className="p-5">
      <div className="mb-5 border-b border-border pb-3">
        <h1 className="text-lg font-semibold tracking-tight text-ink">{title}</h1>
        <p className="mt-1 text-sm text-muted">{description}</p>
      </div>
      <div className="border border-dashed border-border bg-surface p-5 text-sm text-muted">
        Not yet implemented — scheduled for {phase} of the implementation plan.
      </div>
    </div>
  );
}
