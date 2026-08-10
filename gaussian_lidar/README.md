# Gaussian LiDAR service

CLI-сервис загружает Gaussian Splatting PLY через Python API
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
uv run gaussian-lidar /path/to/model.ply --transport websocket --host 127.0.0.1 --port 8765
```

### Отладочный 3D-интерфейс

WebSocket-транспорт также раздаёт исходный PLY по `/scene.ply` и собранный интерфейс по
корневому URL. Интерфейс на Vite, TypeScript, Three.js и Spark показывает сплаты; наведение
на сцену ставит метровый вертикальный маркер, верхняя точка которого используется как позиция
лидара. После небольшого debounce результат сканирования рисуется поверх сцены сферами.

Перед первым запуском соберите frontend:

```bash
cd frontend
npm install
npm run build
cd ..
uv run gaussian-lidar /path/to/model.ply --host 127.0.0.1 --port 8765
```

После этого откройте `http://127.0.0.1:8765/`. Для разработки можно запустить `npm run dev`:
Vite проксирует `/scene.ply` и `/ws` на сервис по адресу `127.0.0.1:8765`.

Для RabbitMQ установите extra и укажите URL/очередь:

```bash
uv sync --extra rabbitmq
uv run gaussian-lidar model.ply --transport rabbitmq \
  --rabbitmq-url amqp://guest:guest@localhost/ --queue gaussian-lidar.scan
```

За декодирование ASCII/binary PLY отвечает `gaussforge.load_ply`; собственный PLY-парсер
в сервисе отсутствует. Адаптер принимает поля позиций `positions`, `means`, `xyz` или
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
