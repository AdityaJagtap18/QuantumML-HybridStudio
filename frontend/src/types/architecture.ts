export type ModelKind = "classical_mlp" | "classical_cnn" | "hybrid_mlp_qnn" | "hybrid_cnn_qnn";
export type Activation = "relu" | "tanh" | "sigmoid";
export type FeatureMapType = "ZFeatureMap" | "ZZFeatureMap";
export type AnsatzType = "RealAmplitudes" | "EfficientSU2";
export type QNNType = "estimator" | "sampler";
export type Entanglement = "linear" | "circular" | "full";

export interface PreprocessingConfig {
  normalize: boolean;
  use_pca: boolean;
  n_components: number | null;
}

export interface MLPExtractorConfig {
  hidden_dims: number[];
  activation: Activation;
  dropout: number;
}

export interface CNNExtractorConfig {
  channels: number[];
  kernel_size: number;
  pooling: boolean;
  fc_dim: number;
  activation: Activation;
  dropout: number;
}

export interface ProjectionConfig {
  activation: Activation;
  normalize: boolean;
}

export interface QuantumLayerConfig {
  n_qubits: number;
  feature_map: FeatureMapType;
  feature_map_reps: number;
  ansatz: AnsatzType;
  ansatz_reps: number;
  entanglement: Entanglement;
  qnn_type: QNNType;
  shots: number;
}

export interface ClassificationHeadConfig {
  n_outputs: number;
}

export interface ArchitectureConfig {
  model_kind: ModelKind;
  mlp_extractor: MLPExtractorConfig;
  cnn_extractor: CNNExtractorConfig;
  projection: ProjectionConfig;
  quantum_layer: QuantumLayerConfig;
  head: ClassificationHeadConfig;
}

export interface CircuitPreviewResponse {
  n_qubits: number;
  depth: number;
  gate_counts: Record<string, number>;
  n_encoding_parameters: number;
  n_trainable_parameters: number;
  ascii_drawing: string;
}
