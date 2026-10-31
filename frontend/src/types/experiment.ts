import type { ArchitectureConfig, PreprocessingConfig } from "./architecture";
import type { DatasetConfigRequest } from "./dataset";
import type { TrainingConfig } from "./training";

export interface ExperimentCreateRequest {
  name: string;
  dataset: DatasetConfigRequest;
  preprocessing: PreprocessingConfig;
  architecture: ArchitectureConfig;
  training: TrainingConfig;
}

export type ExperimentStatus = "QUEUED" | "RUNNING" | "COMPLETED" | "FAILED";
export type ExperimentStage =
  | "QUEUED"
  | "VALIDATING"
  | "PREPROCESSING"
  | "BUILDING_MODEL"
  | "TRAINING"
  | "EVALUATING"
  | "COMPLETED"
  | "FAILED";

export interface ExperimentSummary {
  id: string;
  name: string;
  status: ExperimentStatus;
  stage: ExperimentStage;
  dataset_id: string;
  architecture_kind: string;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  error: string | null;
}

export interface EpochRecord {
  epoch: number;
  train_loss: number;
  val_loss: number;
  train_accuracy: number;
  val_accuracy: number;
  learning_rate: number;
  duration_seconds: number;
}

export interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  balanced_accuracy: number;
  mcc: number;
  roc_auc: number | null;
  pr_auc: number | null;
  confusion_matrix: number[][];
  roc_curve: { fpr: number[]; tpr: number[] } | null;
  pr_curve: { precision: number[]; recall: number[] } | null;
  per_class: Record<string, { precision: number; recall: number; f1: number }>;
}

export interface ImagePredictionResult {
  index: number;
  true_label: number;
  predicted_label: number;
  confidence: number;
  correct: boolean;
  pixels: number[][];
}

export interface ResourceSummary {
  classical_parameters: number;
  quantum_parameters: number;
  total_parameters: number;
  uses_quantum_layer: boolean;
  circuit: {
    n_qubits: number;
    depth: number;
    gate_counts: Record<string, number>;
    n_encoding_parameters: number;
    n_trainable_parameters: number;
    ascii_drawing: string;
  } | null;
}

export interface ExperimentResult {
  dataset: {
    id: string;
    name: string;
    image_height: number;
    image_width: number;
    n_samples: number;
    class_names: string[];
    train_size: number;
    val_size: number;
    test_size: number;
    random_seed: number;
  };
  preprocessing: {
    pipeline: string[];
    original_dim: number;
    final_dim: number;
    flattened: boolean;
  };
  architecture: {
    model_kind: string;
    uses_quantum_layer: boolean;
    uses_cnn: boolean;
  };
  resource_summary: ResourceSummary;
  training: {
    optimizer: string;
    configured_epochs: number;
    stopped_epoch: number;
    early_stopped: boolean;
    best_val_loss: number;
    total_seconds: number;
    random_seed: number;
  };
  metrics: ModelMetrics;
  predictions: ImagePredictionResult[];
  checkpoint_path: string;
  software_versions: Record<string, string>;
}

export interface ExperimentDetail extends ExperimentSummary {
  config: ExperimentCreateRequest;
  result: ExperimentResult | null;
  training_history: EpochRecord[] | null;
}
