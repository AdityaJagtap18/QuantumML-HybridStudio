export type OptimizerType = "adam" | "sgd";

export interface TrainingConfig {
  epochs: number;
  batch_size: number;
  learning_rate: number;
  optimizer: OptimizerType;
  random_seed: number;
  early_stopping_patience: number | null;
}
