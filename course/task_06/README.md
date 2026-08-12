# Задача 6. Показать неизменяемую сцену в Rerun

Ориентируйтесь на готовую реализацию `initialize_rerun` в
`course/reference/02_circular_scan.py`. **Допишите** пять частей: `rr.init`,
blueprint с origin `world`, `ViewCoordinates`, `GaussianSplats3D` и
`LineStrips3D`. `RERUN_UP_AXIS` меняет камеру, но не координаты данных. Полная
реализация уже станет рабочим кодом в `task_07/exercise.py`.

Ключевой готовый фрагмент преобразования:

```python
rgba = torch.cat((rgb.expand(scene.means.shape[0], 3),
                  scene.opacities[:, None]), dim=-1)
rr.GaussianSplats3D(numpy(scene.means), scales=numpy(scene.scales),
    quaternions=numpy(scene.rotations[:, [1, 2, 3, 0]]),
    colors=numpy(rgba * 255).astype("uint8"))
```

Модуль `rerun_viewer` демонстрирует тот же boundary: вычисления остаются в
Torch, и лишь визуализация делает `detach().cpu().numpy()`. Rerun принимает
quaternion `(x,y,z,w)`, поэтому перестановка обязательна. Самопроверка: видны
гауссианы и замкнутое голубое кольцо. Запустите
`uv run python course/task_06/exercise.py`: помимо окна Rerun, `main()` напечатает
число отправленных гауссиан и точек орбиты.
