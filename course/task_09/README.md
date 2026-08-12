# Задача 9. Собрать статическую часть симулятора

Главная оптимизация эксперимента — не загружать сцену и не строить BVH в цикле.
Сначала `scene_cpu`, `bvh_cpu` и статическая визуализация; затем `.to(device)` для
обоих объектов. `GaussianCloud.to` и `FlatBVH.to` возвращают новые dataclass,
сохраняя CPU-оригиналы.

```python
def run_experiment() -> None:
    device = select_device(BACKEND)
    scene_cpu = load_scene(SCENE_PATH)
    bvh_cpu = create_bvh(scene_cpu)
    initialize_rerun(scene_cpu)
    scene, bvh = scene_cpu.to(device), bvh_cpu.to(device)
    simulator = LidarSimulator(scene, bvh, BACKEND)
    config = create_config()
    write_debug_header(scene_cpu, config)
    pose = LidarPose(orbit_position(device, 0.0), lidar_orientation(device))
    scan = simulator.scan(pose, config)
    diagnostics, _ = scan_diagnostics(0, pose, scan, None)
    with DEBUG_LOG_PATH.open("a") as log:
        log.write(json.dumps(diagnostics) + "\n")
    log_scan(0, pose, scan.valid_points())
    print(f"Устройство: {device}; гауссиан: {scene.means.shape[0]}; "
          f"узлов BVH: {bvh.bbox_min.shape[0]}; попаданий: {diagnostics['hits']}")
```

**Допишите** `run_experiment` этим конвейером, создайте позу для угла `0.0`,
выполните ровно один `simulator.scan`, запишите диагностику и вызовите `log_scan`.
`LidarSimulator.scan` делегирует backend, сохраняя единый API. Команда
`uv run python course/task_09/exercise.py` должна открыть первый кадр и напечатать
устройство, размеры сцены/BVH и количество попаданий. В `task_10` это готовое
решение уже расширено до полной статической части финального эксперимента.
