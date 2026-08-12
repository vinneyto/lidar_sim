# Задача 8. Добавить воспроизводимую диагностику

Поля `LidarScan`: `ranges`, `points`, `hit_mask`, `gaussian_ids`,
`accumulated_alpha` и два необязательных overflow-счётчика Metal. Ограниченные
списки кандидатов/стек не должны переполняться молча. Digest помогает отличить
реальное движение от нестабильного backend.

```python
def tensor_digest(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()[:16]
```

Digest уже реализован предыдущим готовым кодом. **Допишите** `scan_diagnostics`.
Алгоритм: detach на CPU; фильтрация
range через hit mask; конечные alpha через `isfinite`; min/mean/max; суммы по
elevation; при наличии предыдущего кадра — XOR масок, смена id и максимальная
дельта общих попаданий. Самопроверка: одна JSON-строка на кадр читается через
`json.loads`, а счётчики overflow видны явно. Команда
`uv run python course/task_08/exercise.py` печатает всю запись первого скана как
форматированный JSON.
