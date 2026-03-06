
from dataclasses import dataclass, field

@dataclass
class Sample:
    image_path: str
    ground_truth: str
    sample_id: str = ""

@dataclass
class ModelResult:
    sample_id: str
    prediction: str
    ground_truth: str
    wer: float
    cer: float
    latency_ms: float
    error: str = ""

@dataclass
class ModelSummary:
    model_name: str
    avg_wer: float
    avg_cer: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    total_samples: int
    failed_samples: int
    results: list[ModelResult] = field(default_factory=list)
