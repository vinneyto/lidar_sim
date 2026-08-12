# Задача 1. Загрузить облако гауссиан

Откройте `exercise.py` и реализуйте `load_scene`. Модуль
`gs_lidar.loader` читает canonical Inria PLY: `scale_*` хранится в log-space,
`opacity` — как logit, а `rot_0..3` — quaternion `(w,x,y,z)`. Загрузчик сам
применяет `exp`/`sigmoid`, извлекает DC-цвет и нормализует вращения. Результат —
неизменяемый `GaussianCloud` с тензорами `[N,3]`, `[N,4]` и `[N]`.

**Допишите:** понятную проверку пути и вызов публичного загрузчика.

```python
def load_scene(path: Path) -> GaussianCloud:
    if not path.is_file():
        raise FileNotFoundError(f"PLY scene not found: {path}. Set PLY_PATH.")
    return load_gaussian_ply(path)
```

Самопроверка: `scene.means.shape == (N, 3)`, масштабы положительны, opacity лежит
в `[0,1]`. Не вызывайте NumPy и не декодируйте PLY повторно. После настройки
`PLY_PATH` запустите `uv run python course/task_01/exercise.py`: `main()` выведет
число гауссиан и диапазон opacity.
