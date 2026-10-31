import { PlaceholderPanel } from "../components/workspace/PlaceholderPanel";

export default function BenchmarkWorkspace() {
  return (
    <PlaceholderPanel
      title="Benchmark Workspace"
      description="Compare classical MLP/CNN against hybrid MLP+QNN/CNN+QNN on accuracy, F1, timing, and parameter count."
      phase="Phase 7 (Benchmarking)"
    />
  );
}
