import {
  Beaker,
  BookOpen,
  GitCompare,
  Images,
  LayoutGrid,
  ListTree,
  Microscope,
  Workflow,
} from "lucide-react";
import { NavLink } from "react-router-dom";

const RAIL_ITEMS = [
  { to: "/", icon: LayoutGrid, label: "Workspace" },
  { to: "/images", icon: Images, label: "Image Explorer" },
  { to: "/architecture", icon: Workflow, label: "Architecture Studio" },
  { to: "/training", icon: Beaker, label: "Training Control Room" },
  { to: "/predictions", icon: Microscope, label: "Prediction Analysis" },
  { to: "/benchmark", icon: GitCompare, label: "Benchmark Workspace" },
  { to: "/library", icon: ListTree, label: "Experiment Library" },
  { to: "/docs", icon: BookOpen, label: "Documentation" },
];

export function ProjectRail() {
  return (
    <nav className="flex h-full w-14 shrink-0 flex-col items-center gap-1 border-r border-border bg-surface py-3">
      {RAIL_ITEMS.map(({ to, icon: Icon, label }) => (
        <NavLink
          key={to}
          to={to}
          end={to === "/"}
          title={label}
          className={({ isActive }) =>
            `group relative flex h-10 w-10 items-center justify-center rounded-sm transition-colors ${
              isActive ? "bg-accent/10 text-accent" : "text-muted hover:bg-canvas hover:text-ink"
            }`
          }
        >
          <Icon size={18} />
        </NavLink>
      ))}
    </nav>
  );
}
