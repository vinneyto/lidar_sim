# Задача 5. Ориентировать шаблон лучей

`lidar.generate_rays` строит endpoint-exclusive азимут и inclusive elevation,
создаёт локальные единичные направления и вращает их quaternion-матрицей. Родной
шаблон Z-up; для Y-up сцены поворот `-90°` вокруг X кладёт азимутальную плоскость
в XZ. Quaternion в проекте всегда `(w,x,y,z)`.

```python
def lidar_orientation(device):
    half_angle = -math.pi / 4
    return torch.tensor([math.cos(half_angle), math.sin(half_angle), 0., 0.],
                        dtype=torch.float32, device=device)

def create_config():
    return LidarConfig(azimuth_samples=AZIMUTH_SAMPLES,
        elevation_samples=ELEVATION_SAMPLES, near=NEAR, far=FAR,
        gaussian_sigma_cutoff=SIGMA_CUTOFF,
        accumulated_alpha_threshold=ALPHA_THRESHOLD)
```

**Допишите** обе функции. Порог alpha означает первый пик, после которого
front-to-back композиция `A ← A + (1-A)α` достигнет значения, а не пересечение
границы 3σ. Команда `uv run python course/task_05/exercise.py` выводит число
сгенерированных лучей, их общее начало и норму quaternion.
