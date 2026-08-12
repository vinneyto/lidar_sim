# Задача 7. Записать один кадр сканирования

`LidarScan.valid_points()` применяет `hit_mask` к `[E,A,3]` и возвращает `[M,3]`.
Промахи (`nan`, `inf`, id `-1`) в viewer передавать не нужно. В Rerun каждый
шаг помещается на sequence timeline `scan`.

```python
def log_scan(step, pose, points):
    import rerun as rr
    rr.set_time("scan", sequence=step)
    rr.log("world/lidar/position", rr.Points3D(
        pose.position.detach().cpu().numpy()[None],
        colors=[255, 180, 0], radii=0.08))
    rr.log("world/lidar/returns", rr.Points3D(
        points.detach().cpu().numpy(), colors=[0, 255, 120], radii=0.01))
```

**Допишите** функцию и используйте актуальный `set_time(..., sequence=...)`.
Самопроверка: ползунок `scan` переключает облака, а не накапливает их в одном
статическом кадре. `uv run python course/task_07/exercise.py` выполняет настоящий
первый скан, отправляет его в Rerun и печатает число попаданий.
