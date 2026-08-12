# Задача 10. Замкнуть цикл эксперимента

Угловая скорость задаётся в градусах/с, но `sin`/`cos` принимают радианы.
На шаге `k`: `angle = radians(speed) * STEP_SECONDS * k`; далее создаются
`LidarPose`, скан и диагностика. Между кадрами (кроме последнего) выполняется
`sleep`, чтобы движение соответствовало модельному времени.

```python
for step in range(NUMBER_OF_STEPS):
    angle = angular_velocity * STEP_SECONDS * step
    pose = LidarPose(orbit_position(device, angle), orientation)
    scan = simulator.scan(pose, config)
    diagnostics, previous = scan_diagnostics(step, pose, scan, previous)
    with DEBUG_LOG_PATH.open("a") as log:
        log.write(json.dumps(diagnostics) + "\n")
    log_scan(step, pose, scan.valid_points())
    if step + 1 < NUMBER_OF_STEPS:
        time.sleep(STEP_SECONDS)
```

**Допишите** цикл и информативный `print`. Функция `main()` и guard уже находятся
внизу файла. Запустите
`uv run python course/task_10/exercise.py` и сверите файл с reference: после
замены единственного `TODO` это и есть второй эксперимент.
Итоговая ручная проверка: sensor идёт по кольцу, timeline содержит заданное число
кадров, возвраты меняются плавно, JSONL содержит header и запись на каждый шаг.
