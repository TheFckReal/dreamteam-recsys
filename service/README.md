# Запуск MLflow-инфраструктуры

Этот `docker-compose.yml` поднимает локальную инфраструктуру для MLflow:

- `mlflow` - tracking server и UI;
- `s3` - локальное S3-хранилище на MinIO;
- `postgres` - база для метаданных MLflow;
- `create-bucket` - одноразовый init-контейнер, который создаёт bucket в MinIO.

## Требования

- Windows 10/11.
- Установленный Docker Desktop (https://docs.docker.com/desktop/setup/install/windows-install/).
- Запущенный Docker Desktop перед выполнением команд.
- Файл `.env` в корне проекта.

Проверить Docker:

```powershell
docker --version
docker compose version
```

## Переменные окружения

В корневом `.env` должны быть заданы:

```env
MINIO_ROOT_USER=minio_adm
MINIO_ROOT_PASSWORD=minio_password_123
DEFAULT_BUCKET_NAME=mlflow

POSTGRES_USER=mlflow
POSTGRES_PASSWORD=mlflow_password_123
POSTGRES_DB=mlflow
```

Пароли выше приведены как пример. Для реального использования замените их на свои.

## Запуск

Из корня проекта:

```powershell
docker compose --env-file .env -f service/docker-compose.yml up
```

Или из папки `service`:

```powershell
docker compose --env-file ../.env up
```

Для запуска в фоне добавьте `-d`:

```powershell
docker compose --env-file ../.env up -d
```

## Доступные UI

- MLflow UI: http://localhost:5050
- MinIO UI: http://localhost:9001

Для входа в MinIO используйте `MINIO_ROOT_USER` и `MINIO_ROOT_PASSWORD` из `.env`.

## Остановка

Из папки `service`:

```powershell
docker compose --env-file ../.env down
```

Команда `down` не удаляет данные из volumes. Чтобы удалить все данные PostgreSQL, MinIO и MLflow, используйте:

```powershell
docker compose --env-file ../.env down -v
```

Используйте `down -v` осторожно: он удалит bucket MinIO, PostgreSQL metadata и локальные данные MLflow.
