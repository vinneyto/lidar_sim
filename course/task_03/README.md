# Задача 3. Выбрать backend и устройство

`LidarSimulator` выбирает `CpuLidarTracer` для CPU и `MetalLidarTracer` для MPS.
CPU — читаемая эталонная реализация. Metal запускает один shader-thread на луч и
ожидает, что сцена и BVH уже находятся на `mps`; CUDA проект не использует.

```python
def select_device(backend: str) -> torch.device:
    if backend == "cpu":
        return torch.device("cpu")
    if backend != "metal":
        raise ValueError("BACKEND must be either 'cpu' or 'metal'")
    if not torch.backends.mps.is_available():
        raise RuntimeError("BACKEND='metal' requires PyTorch MPS")
    return torch.device("mps")
```

**Допишите** развилку без молчаливого fallback: иначе пользователь может считать,
что измеряет Metal, хотя работает CPU. Самопроверка: `cpu` работает на любой ОС,
опечатка завершается `ValueError`. Команда
`uv run python course/task_03/exercise.py` печатает выбранные backend/device и
подтверждает, что BVH строился на CPU.
