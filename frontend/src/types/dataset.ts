export type DatasetId = "digits" | "mnist_subset" | "fashion_mnist_subset";

export interface DatasetInfo {
  id: DatasetId;
  name: string;
  description: string;
  image_height: number;
  image_width: number;
  channels: number;
  n_samples: number;
  n_classes: number;
  class_names: string[];
  available: boolean;
  unavailable_reason: string | null;
}

export interface DatasetConfigRequest {
  dataset_id: DatasetId;
  selected_classes?: number[] | null;
  max_samples?: number | null;
  test_size: number;
  val_size: number;
  random_seed: number;
}

export interface ClassDistributionEntry {
  label: number;
  class_name: string;
  count: number;
}

export interface ImageSample {
  index: number;
  label: number;
  class_name: string;
  pixels: number[][];
}

export interface DatasetPreviewResponse {
  dataset: DatasetInfo;
  total_samples: number;
  train_samples: number;
  val_samples: number;
  test_samples: number;
  class_distribution: ClassDistributionEntry[];
  sample_images: ImageSample[];
  pixel_min: number;
  pixel_max: number;
  pixel_mean: number;
}
