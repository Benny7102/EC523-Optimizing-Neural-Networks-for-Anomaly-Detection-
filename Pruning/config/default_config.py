from dataclasses import dataclass

@dataclass
class TrainingConfig:
    lr: float = 1.2e-3
    batch_size: int = 60
    start_iter: int = 0
    end_iter: int = 100
    print_freq: int = 1
    valid_freq: int = 1

@dataclass
class PruningConfig:
    prune_type: str = "lt"  # lt | reinit
    prune_percent: int = 10
    prune_iterations: int = 35

@dataclass
class DefaultConfig:
    training: TrainingConfig = TrainingConfig()
    pruning: PruningConfig = PruningConfig()
    gpu: str = "0"
    dataset: str = "mnist"
    arch_type: str = "fc1"
    resume: bool = False