# Задача 1. Загрузить облако гауссиан

Откройте `exercise.py` и реализуйте `load_scene`. Модуль
`gs_lidar.loader` использует `gsply` и поэтому читает PLY, компактный SOG и
другие поддерживаемые библиотекой форматы. `gsply` возвращает единое
представление: линейные масштабы, opacity в `[0,1]`, quaternion `(w,x,y,z)`.
Результат —
неизменяемый `GaussianCloud` с тензорами `[N,3]`, `[N,4]` и `[N]`.

**Допишите:** понятную проверку пути и вызов публичного загрузчика.

```python
def load_scene(path: Path) -> GaussianCloud:
    if not path.is_file():
        raise FileNotFoundError(f"Gaussian scene not found: {path}. Set SCENE_PATH.")
    return load_gaussian_scene(path)
```

Самопроверка: `scene.means.shape == (N, 3)`, масштабы положительны, opacity лежит
в `[0,1]`. Не вызывайте NumPy и не декодируйте файл сцены повторно. После настройки
`SCENE_PATH` запустите `uv run python course/task_01/exercise.py`: `main()` выведет
число гауссиан и диапазон opacity.
