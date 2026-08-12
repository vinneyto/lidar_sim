# Задача 4. Описать круговую траекторию

Поза LiDAR состоит из позиции и quaternion. Пока создайте позиции окружности в
XZ: `x=cx+r*cos(a)`, `y=cy`, `z=cz+r*sin(a)`. Полилиния нужна только зрителю;
последняя точка должна повторить первую, поэтому `linspace` включает `2π`.

```python
def orbit_position(device, angle_radians):
    center = torch.tensor(ORBIT_CENTER, dtype=torch.float32, device=device)
    offset = center.new_tensor([ORBIT_RADIUS * math.cos(angle_radians), 0.0,
                                ORBIT_RADIUS * math.sin(angle_radians)])
    return center + offset

def orbit_points():
    center = torch.tensor(ORBIT_CENTER, dtype=torch.float32)
    angles = torch.linspace(0, 2 * math.pi, RING_SAMPLES + 1)
    points = center.expand(RING_SAMPLES + 1, 3).clone()
    points[:, 0] += ORBIT_RADIUS * torch.cos(angles)
    points[:, 2] += ORBIT_RADIUS * torch.sin(angles)
    return points
```

**Допишите** обе функции. Самопроверка: нормы XZ-смещений равны радиусу, а
`points[0]` и `points[-1]` совпадают. Запустите
`uv run python course/task_04/exercise.py`, чтобы увидеть три опорные позиции и
результат проверки замкнутости кольца.
