import { Route, Routes } from "react-router-dom";

import { ProjectRail } from "./components/workspace/ProjectRail";
import { TopBar } from "./components/workspace/TopBar";
import ArchitectureStudio from "./pages/ArchitectureStudio";
import BenchmarkWorkspace from "./pages/BenchmarkWorkspace";
import Documentation from "./pages/Documentation";
import ExperimentLibrary from "./pages/ExperimentLibrary";
import ImageExplorer from "./pages/ImageExplorer";
import PredictionAnalysis from "./pages/PredictionAnalysis";
import TrainingControlRoom from "./pages/TrainingControlRoom";
import Workspace from "./pages/Workspace";

export default function App() {
  return (
    <div className="flex h-screen flex-col bg-canvas">
      <TopBar />
      <div className="flex min-h-0 flex-1">
        <ProjectRail />
        <main className="min-w-0 flex-1 overflow-y-auto">
          <Routes>
            <Route path="/" element={<Workspace />} />
            <Route path="/images" element={<ImageExplorer />} />
            <Route path="/architecture" element={<ArchitectureStudio />} />
            <Route path="/training" element={<TrainingControlRoom />} />
            <Route path="/training/:id" element={<TrainingControlRoom />} />
            <Route path="/predictions" element={<PredictionAnalysis />} />
            <Route path="/predictions/:id" element={<PredictionAnalysis />} />
            <Route path="/benchmark" element={<BenchmarkWorkspace />} />
            <Route path="/library" element={<ExperimentLibrary />} />
            <Route path="/docs" element={<Documentation />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}
