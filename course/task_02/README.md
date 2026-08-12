# Задача 2. Построить ускоряющую структуру

`gaussian_geometry.gaussian_aabbs` переводит quaternion в матрицу и вычисляет
осевой bounding box эллипсоида на расстоянии `sigma_cutoff`. Это лишь конечная
область кандидатов, а не поверхность возврата. `bvh.build_bvh` на CPU рекурсивно
делит примитивы медианой по оси максимального разброса и упаковывает дерево в
тензоры `FlatBVH`, удобные CPU- и Metal-трассировщикам.

```python
def create_bvh(scene: GaussianCloud) -> FlatBVH:
    bbox_min, bbox_max = gaussian_aabbs(
        scene.means, scene.scales, scene.rotations, SIGMA_CUTOFF
    )
    return build_bvh(bbox_min, bbox_max)
```

**Допишите** оба вызова. Стройте BVH ровно один раз до переноса сцены на MPS.
Самопроверка: корневой bbox покрывает все примитивы, `primitive_indices` содержит
ровно `N` индексов. Запустите `uv run python course/task_02/exercise.py` — в
консоли появятся количества гауссиан, узлов BVH и примитивов в листьях.
