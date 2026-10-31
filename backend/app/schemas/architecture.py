from typing import Literal

from pydantic import BaseModel, Field

ModelKind = Literal["classical_mlp", "classical_cnn", "hybrid_mlp_qnn", "hybrid_cnn_qnn"]
Activation = Literal["relu", "tanh", "sigmoid"]
FeatureMapType = Literal["ZFeatureMap", "ZZFeatureMap"]
AnsatzType = Literal["RealAmplitudes", "EfficientSU2"]
QNNType = Literal["estimator", "sampler"]
Entanglement = Literal["linear", "circular", "full"]


class PreprocessingConfig(BaseModel):
    normalize: bool = True
    use_pca: bool = False
    n_components: int | None = None


class MLPExtractorConfig(BaseModel):
    hidden_dims: list[int] = [32, 16]
    activation: Activation = "relu"
    dropout: float = Field(default=0.0, ge=0.0, le=0.9)


class CNNExtractorConfig(BaseModel):
    channels: list[int] = [8, 16]
    kernel_size: int = Field(default=3, ge=2, le=5)
    pooling: bool = True
    fc_dim: int = 32
    activation: Activation = "relu"
    dropout: float = Field(default=0.0, ge=0.0, le=0.9)


class ProjectionConfig(BaseModel):
    activation: Activation = "tanh"
    normalize: bool = True


class QuantumLayerConfig(BaseModel):
    n_qubits: int = Field(ge=1, le=6)
    feature_map: FeatureMapType = "ZZFeatureMap"
    feature_map_reps: int = Field(default=1, ge=1, le=3)
    ansatz: AnsatzType = "RealAmplitudes"
    ansatz_reps: int = Field(default=2, ge=1, le=5)
    entanglement: Entanglement = "linear"
    qnn_type: QNNType = "estimator"
    shots: int = Field(default=1024, ge=16, le=20000)


class ClassificationHeadConfig(BaseModel):
    n_outputs: int = 1


class ArchitectureConfig(BaseModel):
    model_kind: ModelKind
    mlp_extractor: MLPExtractorConfig = MLPExtractorConfig()
    cnn_extractor: CNNExtractorConfig = CNNExtractorConfig()
    projection: ProjectionConfig = ProjectionConfig()
    quantum_layer: QuantumLayerConfig = QuantumLayerConfig(n_qubits=4)
    head: ClassificationHeadConfig = ClassificationHeadConfig()

    @property
    def uses_quantum_layer(self) -> bool:
        return self.model_kind in ("hybrid_mlp_qnn", "hybrid_cnn_qnn")

    @property
    def uses_cnn(self) -> bool:
        return self.model_kind in ("classical_cnn", "hybrid_cnn_qnn")
