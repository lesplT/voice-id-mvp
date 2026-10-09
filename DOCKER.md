# Docker: весь Voice ID MVP

Запуск CPU-версии на Windows с Docker Desktop (Linux containers) или Linux x64.
На новом компьютере нужен интернет для скачивания образов, Python-пакетов и готовых моделей.
Локальные Python, FFmpeg и виртуальное окружение устанавливать не требуется.

```powershell
git clone https://github.com/lesplT/voice-id-mvp.git
cd voice-id-mvp
docker compose -f compose.docker.yml up -d --build --wait --wait-timeout 1800
```

Откройте **http://127.0.0.1:18000/**. Приватный репозиторий требует входа в GitHub.
Первый запуск скачивает модели; это не обучение с нуля и может занять несколько минут.
Модели не входят в GitHub и в образ. Если папка `models` уже содержит рабочие веса,
инициализация использует их повторно, проверяя ECAPA и загружая GigaAM.

Структура: отдельные workspace, enrollment, identification и transcription,
Qdrant и завершающийся после подготовки сервис models. Workspace ждёт готовности
сервисов обработки. Только страница 18000 открыта на localhost; Qdrant и API внутри сети Compose.
Linux CPU PyTorch/torchaudio закреплены на одинаковой версии 2.6.0, GigaAM — на конкретном commit.
Windows-файл `requirements.freeze.txt` не используется как Linux lock.
FFmpeg устанавливается из готового `imageio-ffmpeg`, без apt и без установки на Windows.
При ошибке загрузки GigaAM инициализация останавливается, а не заявляет успешный запуск
с незаметно отключённой основной моделью. Во время обработки доступен прежний Vosk fallback.

## Данные и остановка

- `models/`: скачанные веса, не личные записи.
- `data/docker/workspace/journal/`: журнал, оригинальное аудио и результаты.
- `data/docker/workspace/dictionary/`: словарь и аудиопримеры.
- `data/docker/qdrant/`: зарегистрированные голосовые отпечатки.
- `data/docker/model-cache/`: служебный кеш инициализации.

Все эти папки исключены из GitHub и контекста сборки Docker. Журнал хранится до ручного удаления.
Native-данные в `data/journal`, `data/dictionary`, `data/qdrant_local` не перезаписываются;
Docker использует отдельную копию. Последующие изменения в двух версиях не синхронизируются.

```powershell
docker compose -f compose.docker.yml ps
docker compose -f compose.docker.yml logs --tail 80 models workspace transcription
docker compose -f compose.docker.yml stop
docker compose -f compose.docker.yml start --wait
```

`stop` сохраняет контейнеры и данные. `down` удаляет контейнеры/сеть, но перечисленные
папки остаются на диске. Не удаляйте `data/docker` для «очистки»: там личные записи.
Для резервной копии остановите Compose, затем скопируйте `data/docker` и при необходимости `models`.
Данные не зашифрованы. Приватный GitHub не является резервной копией журнала.

Для другого порта в PowerShell: `$env:VOICE_ID_PORT='18001'`, затем команда `up` выше.
Для остановки старых Windows-сервисов можно использовать `scripts/stop_all.ps1`;
он не останавливает новую Compose-систему. Не запускайте `start_all.ps1` для запуска Docker-версии.

## Перенос существующих голосов и журнала с Windows

Только **до первого запуска Docker**, пока целевой `data/docker/workspace` пуст.
Дождитесь завершения задач в старом интерфейсе; не загружайте записи во время снимка.

```powershell
.\.venv\Scripts\python.exe scripts\prepare_docker_data.py
docker compose -f compose.docker.yml up -d --build --wait --wait-timeout 1800
Get-Content data\docker\voice-migration.json -Raw | docker compose -f compose.docker.yml exec -T enrollment python scripts/docker_import_voices.py
```

Скрипт делает SQLite-снимки журнала/словаря, копирует аудио и экспортирует голосовые
векторы в локальный `data/docker/voice-migration.json`. Он читает native embedded Qdrant;
если старая база работала через отдельный сервер, нужен отдельный сценарий переноса.
Импорт требует пустую целевую коллекцию и отказывается перезаписывать существующие голоса.
Файл экспорта содержит личные голосовые данные, не публикуйте его.
В этом проекте перенос сделан один раз; повторная команда намеренно выдаст отказ.

## Проверки

```powershell
docker compose -f compose.docker.yml config --quiet
docker build --target test -t voice-id-mvp:test .
docker run --rm voice-id-mvp:test
```

GitHub Actions повторяет в Linux проверку состава репозитория, Compose и тесты контейнера.
Личные аудиозаписи не отправляются в Actions; реальные проверки запускайте локально.
Основной интерфейс позволяет проверить все функции без обращения к скрытым портам.
