# Gaussian LiDAR service

CLI-сервис скачивает Gaussian Splatting PLY по HTTP(S) URL в память и разбирает его через Python API
[GaussForge](https://github.com/3dgscloud/GaussForge), строит плоское BVH в PyTorch и на macOS
трассирует лидарные лучи Metal compute-ядром, интегрированным непосредственно через
`torch.mps.compile_shader`. Сцена, BVH, лучи и результаты представлены PyTorch tensors;
отдельной PyObjC/C++-обёртки нет. Транспорт выбирается при запуске: WebSocket или
RabbitMQ RPC. Проект намеренно находится в отдельной директории.

## Требования и запуск

- macOS и GPU с поддержкой Metal/MPS;
- Python 3.11+;
- PyTorch 2.7+ со сборкой, содержащей `torch.mps.compile_shader`;
- [uv](https://docs.astral.sh/uv/).

```bash
cd gaussian_lidar
uv sync
uv run gaussian-lidar https://assets.example/model.ply \
  --transport websocket --host 127.0.0.1 --port 8765
```

Для RabbitMQ установите extra и укажите URL/очередь:

```bash
uv sync --extra rabbitmq
uv run gaussian-lidar https://assets.example/model.ply --transport rabbitmq \
  --rabbitmq-url amqp://guest:guest@localhost/ --queue gaussian-lidar.scan
```

При старте сервис скачивает URL с учетом редиректов, не записывает PLY на диск и передает
полученные `bytes` напрямую в `gaussforge.load_ply`. По умолчанию действуют timeout 60 секунд
и лимит 2048 MiB; они меняются через `--download-timeout` и `--max-download-mib`.
За декодирование ASCII/binary PLY отвечает GaussForge; собственный PLY-парсер в сервисе
отсутствует. Адаптер принимает поля позиций `positions`, `means`, `xyz` или
`centers`, масштабы `scales`/`scaling` либо логарифмические `log_scales`/`log_scaling`,
и вращения `rotations`, `quaternions` или `quats`. У обычного point-cloud PLY масштаб
задается `--default-scale`, а единичное вращение добавляется автоматически. Поверхность гауссиана
для пересечения — эллипсоид `--sigma-extent` (по умолчанию 3 sigma). BVH использует
консервативные сферические bounds, поэтому поворот эллипсоида их не нарушает.

После чтения PLY данные становятся CPU tensors. BVH также строится операциями PyTorch,
после чего неизменяемые tensors сцены и дерева один раз переносятся на `mps`. Для каждого
запроса направления лучей генерируются в PyTorch, Metal entry point вызывается как custom
PyTorch operation, а на CPU возвращаются только готовые попадания для сериализации.

## Протокол

Каждое WebSocket-сообщение — JSON-запрос; ответ приходит в то же соединение.
RabbitMQ использует стандартный RPC pattern: запрос должен иметь `reply_to`, а
`correlation_id` переносится в ответ.

```json
{
  "sensor": {
    "position": [0.0, 0.0, 1.8],
    "quaternion": [1.0, 0.0, 0.0, 0.0]
  },
  "scan": {
    "width": 1024,
    "height": 64,
    "horizontal_fov_deg": 360.0,
    "vertical_fov_deg": 30.0,
    "max_distance": 200.0
  }
}
```

Кватернион позы тоже задан как `[w,x,y,z]`; локальная ось `+X` смотрит вперед,
`+Z` — вверх. Лучи идут построчно. Успешный ответ содержит только попадания:

```json
{
  "ok": true,
  "width": 1024,
  "height": 64,
  "points": [[12.4, -1.2, 0.8]],
  "distances": [12.48],
  "ray_indices": [32801]
}
```

`ray_indices` связывает разреженный результат с пикселем:
`row = index // width`, `column = index % width`. Ошибка в запросе возвращается как
`{"ok":false,"error":"..."}`. Одно сообщение ограничено 4 194 304 лучами.

## Разработка

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```
