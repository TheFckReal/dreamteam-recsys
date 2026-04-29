"""Пайплайн feature engineering для данных marketplace домена.

Пайплайн организован как datamart со следующими слоями, все артефакты
сохраняются в :data:`PROCESSED_DATA_DIR`:

* ``datamart/raw/events/{action_type}/{day}.pq`` --
  сырые события, партиционированные по типу действия и дню.
* ``datamart/aggs/events/{day}.pq`` --
  дневные счетчики по парам (user, item) в разрезе action_type и subdomain.
* ``datamart/features/events/{group}/{day}.pq`` --
  признаки на скользящем окне для разных группировок сущностей (``user_id``,
  ``item_id``, ``user_id-item_id``, ``socdem_cluster-brand_id`` и т.д.).
* ``basis/{day}.pq`` --
  ранжировочная основа ``(session_id, day, user_id, item_id, label)`` для каждого
  дня. Колонка ``day`` совместима с :func:`src.dataset.global_temporal_split`.
* ``dataset/{day}.pq`` --
  basis, склеенный с атрибутами user/item и признаками предыдущего дня.

Запуск через Typer CLI, например::

    python -m src.features all
    python -m src.features build-aggs --day-from 1250 --day-to 1300
"""

from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path
import typing as t

from loguru import logger
import polars as pl
from tqdm import tqdm
import typer

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.dataset import create_target

app = typer.Typer(help="Пайплайн feature engineering для данных marketplace.")


# ---------------------------------------------------------------------------
# Константы и пути
# ---------------------------------------------------------------------------

ACTION_TYPES: list[str] = ["view", "click", "clickout", "like"]
"""События по возрастанию бизнес-ценности: ``view < click < clickout < like``."""

SUBDOMAINS: list[str] = ["u2i", "i2i", "catalog", "search", "other"]
"""Все возможные значения поля ``subdomain`` в событиях marketplace."""

CONVERSION_PAIRS: list[tuple[str, str]] = [
    ("view", "click"),
    ("view", "clickout"),
    ("view", "like"),
    ("click", "clickout"),
    ("click", "like"),
    ("clickout", "like"),
]
"""Пары ``(слабое_действие, сильное_действие)`` для расчета конверсионных
метрик вида ``rate_{strong}_given_{weak}``."""

# --- входные (raw) пути ------------------------------------------------------
MARKETPLACE_DIR = RAW_DATA_DIR / "dataset" / "small" / "marketplace"
EVENTS_RAW_DIR = MARKETPLACE_DIR / "events"
ITEMS_RAW_PATH = MARKETPLACE_DIR / "items.pq"
USERS_RAW_PATH = RAW_DATA_DIR / "dataset" / "small" / "users.pq"

# --- выходные (processed) пути -----------------------------------------------
DATAMART_DIR = PROCESSED_DATA_DIR / "datamart"
RAW_LAYER_DIR = DATAMART_DIR / "raw" / "events"
AGG_LAYER_DIR = DATAMART_DIR / "aggs" / "events"
FEATURE_LAYER_DIR = DATAMART_DIR / "features" / "events"

BASIS_DIR = PROCESSED_DATA_DIR / "basis"
DATASET_DIR = PROCESSED_DATA_DIR / "dataset"

SELECTED_USERS_PATH = PROCESSED_DATA_DIR / "selected_users.pq"
SELECTED_ITEMS_PATH = PROCESSED_DATA_DIR / "selected_items.pq"

# По умолчанию диапазон дней не задан и определяется автоматически по
# содержимому соответствующего слоя (raw events / agg / basis / ...).
DEFAULT_DAY_FROM: int | None = None
DEFAULT_DAY_TO: int | None = None

UserGroup = t.Literal["user_id", "region", "socdem_cluster"]
"""Допустимые ключи группировки по стороне пользователя."""

ItemGroup = t.Literal["item_id", "brand_id", "category", "subcategory"]
"""Допустимые ключи группировки по стороне айтема."""


TargetType = t.Literal["log_target", "sqrt_target", "unproccessed", "multiclass"]
"""Допустимые режимы расчета таргета, передаваемые в :func:`src.dataset.create_target`."""

DEFAULT_TARGET_TYPE: TargetType = "log_target"
"""Режим таргета по умолчанию для CLI-команд и функций пайплайна."""

_MULTICLASS_TARGET_COLS = (
    "target_view",
    "target_clickout",
    "target_like",
    "target_click",
)
"""Имена бинарных колонок таргета в режиме ``"multiclass"``."""

DEFAULT_FEATURE_GROUPS: list[tuple[UserGroup | None, ItemGroup | None]] = [
    ("user_id", None),
    ("user_id", "item_id"),
    ("user_id", "brand_id"),
    ("user_id", "category"),
    (None, "item_id"),
    (None, "brand_id"),
    (None, "category"),
    ("socdem_cluster", "item_id"),
    ("socdem_cluster", "brand_id"),
    ("socdem_cluster", "category"),
]


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


def _event_file(day: int, events_dir: Path = EVENTS_RAW_DIR) -> Path:
    """Возвращает путь к файлу сырых событий marketplace для заданного дня.

    Parameters
    ----------
    day : int
        Целочисленный номер дня (используется как имя файла ``{day:05d}.pq``).
    events_dir : Path
        Директория с файлами событий marketplace.

    Returns
    -------
    Path
        Путь вида ``{events_dir}/{day:05d}.pq``.
    """
    return events_dir / f"{day:05d}.pq"


def _existing_event_files(days: Iterable[int], events_dir: Path = EVENTS_RAW_DIR) -> list[Path]:
    """Возвращает список существующих на диске файлов событий marketplace.

    Parameters
    ----------
    days : Iterable[int]
        Итерируемый набор номеров дней для проверки.
    events_dir : Path
        Директория с файлами событий marketplace.

    Returns
    -------
    list[Path]
        Список путей к файлам, которые реально существуют на диске.
        Файлы отсутствующих дней пропускаются без ошибок.
    """
    return [p for p in (_event_file(d, events_dir) for d in days) if p.exists()]


def _discover_day_range(directory: Path, pattern: str = "*.pq") -> tuple[int, int]:
    """Определяет минимальный и максимальный день по parquet-файлам в директории.

    Имена файлов вида ``{day}.pq`` или ``{day:05d}.pq`` (при ``pattern="*/*.pq"``
    — также вложенные ``{action}/{day}.pq``). Файлы с нечисловым stem
    игнорируются.

    Parameters
    ----------
    directory : Path
        Директория, в которой ищутся parquet-файлы с дневной нумерацией.
    pattern : str
        Glob-паттерн для обхода файлов (по умолчанию ``"*.pq"``).
        Для вложенных структур вида ``{action}/{day}.pq`` используйте
        ``"*/*.pq"``.

    Returns
    -------
    tuple[int, int]
        Кортеж ``(min_day, max_day)`` — минимальный и максимальный день из
        найденных файлов.

    Raises
    ------
    FileNotFoundError
        Если в ``directory`` не найдено ни одного файла, соответствующего
        ``pattern`` с числовым именем.
    """
    days: list[int] = []
    for p in directory.glob(pattern):
        try:
            days.append(int(p.stem))
        except ValueError:
            continue
    if not days:
        raise FileNotFoundError(
            f"В {directory} не найдено parquet-файлов с дневной нумерацией (паттерн '{pattern}')."
        )
    return min(days), max(days)


def _resolve_day_range(
    day_from: int | None,
    day_to: int | None,
    directory: Path,
    pattern: str = "*.pq",
) -> tuple[int, int]:
    """Подставляет недостающие границы диапазона по содержимому ``directory``.

    Если ``day_from`` или ``day_to`` равны ``None``, на их место подставляется
    минимальный/максимальный день, найденный в ``directory`` через
    :func:`_discover_day_range`.

    Parameters
    ----------
    day_from : int or None
        Начальный день диапазона или ``None`` для автодетекта.
    day_to : int or None
        Конечный день диапазона или ``None`` для автодетекта.
    directory : Path
        Директория, из которой автодетектятся границы при ``None``.
    pattern : str
        Glob-паттерн для :func:`_discover_day_range`.

    Returns
    -------
    tuple[int, int]
        Кортеж ``(day_from, day_to)`` с заполненными значениями.

    Raises
    ------
    FileNotFoundError
        Если хотя бы одна из границ равна ``None`` и в ``directory`` не
        найдено подходящих файлов.
    """
    if day_from is not None and day_to is not None:
        return day_from, day_to
    auto_from, auto_to = _discover_day_range(directory, pattern)
    return (
        auto_from if day_from is None else day_from,
        auto_to if day_to is None else day_to,
    )


def _feature_name(
    feature: str,
    keys: list[str],
    type: t.Literal["num", "cat"] = "num",
    num_days: int | None = None,
) -> str:
    """Формирует каноническое имя колонки признака.

    Формат: ``f_{type}__{key1}_{key2}__{feature}_{N}d``. Префикс ``f_num__`` /
    ``f_cat__`` используется дальше при разделении признаков на числовые и
    категориальные.

    Parameters
    ----------
    feature : str
        Базовое имя признака (например, ``"mean_num_view_u2i"``).
    keys : list[str]
        Список ключей группировки (например, ``["user_id", "item_id"]``).
    type : {"num", "cat"}
        Тип признака — ``"num"`` для числового, ``"cat"`` для категориального.
    num_days : int or None
        Размер скользящего окна в днях; если ``None`` — суффикс ``_{N}d``
        не добавляется.

    Returns
    -------
    str
        Строка вида ``f_num__user_id_item_id__mean_num_view_u2i_30d``.
    """
    suffix = f"_{num_days}d" if num_days is not None else ""
    return f"f_{type}__{'_'.join(keys)}__{feature}{suffix}"


def _load_features_to_use(filepath: Path | None) -> list[str] | None:
    """Читает список имен папок-признаков из JSON или текстового файла.

    Parameters
    ----------
    filepath : Path or None
        Путь к файлу со списком групп признаков. Поддерживаются форматы:
        ``.json`` (массив строк) и plain-text (одна группа на строку).
        Если ``None``, возвращается ``None`` — сигнал использовать все
        доступные группы.

    Returns
    -------
    list[str] or None
        Список имен папок-групп признаков или ``None``, если ``filepath``
        не задан.
    """
    if filepath is None:
        return None
    filepath = Path(filepath)
    text = filepath.read_text(encoding="utf-8")
    if filepath.suffix == ".json":
        return list(json.loads(text))
    return [line.strip() for line in text.splitlines() if line.strip()]


def _attribute_rename_map(lf: pl.LazyFrame, key: str) -> tuple[dict[str, str], list[str]]:
    """Строит карту переименований колонок-атрибутов user/item в признаки
    ``f_num__`` / ``f_cat__`` и возвращает список категориальных имен.

    Float-колонки получают префикс ``f_num__``, integer/string/categorical —
    ``f_cat__``. Колонка-ключ ``key`` (``user_id`` или ``item_id``) пропускается.

    Parameters
    ----------
    lf : polars.LazyFrame
        LazyFrame с атрибутами пользователей или айтемов.
    key : str
        Имя колонки-идентификатора (``"user_id"`` или ``"item_id"``),
        которая не переименовывается.

    Returns
    -------
    tuple[dict[str, str], list[str]]
        Кортеж ``(rename_map, cat_cols)``:

        * ``rename_map`` — словарь ``{старое_имя: новое_имя}`` для
          :meth:`polars.LazyFrame.rename`;
        * ``cat_cols`` — список новых имен категориальных колонок
          (с префиксом ``f_cat__``).
    """
    rename: dict[str, str] = {}
    cat_cols: list[str] = []
    for col, dtype in lf.collect_schema().items():
        if col == key:
            continue
        if dtype.is_float():
            rename[col] = f"f_num__{col}"
        elif dtype.is_integer() or dtype in (pl.String, pl.Categorical):
            new_name = f"f_cat__{col}"
            rename[col] = new_name
            cat_cols.append(new_name)
    return rename, cat_cols


def _read_id_list(path: Path, column: str) -> list:
    """Читает однострочный parquet с одной колонкой в Python-список.

    Parameters
    ----------
    path : Path
        Путь к parquet-файлу, созданному :func:`select_top_entities`.
    column : str
        Имя колонки для извлечения (``"user_id"`` или ``"item_id"``).

    Returns
    -------
    list
        Python-список значений из указанной колонки.
    """
    return pl.scan_parquet(path).select(column).collect()[column].to_list()


def _compact_dtypes(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Сжимает числовые dtypes до 32-битных там, где это безопасно.

    Float64 → Float32 — для рекомендательных моделей точности более чем
    достаточно, а размер на диске и в памяти уменьшается вдвое. Целочисленные
    типы (``UInt64`` / ``Int64`` / ``UInt32``) намеренно не трогаем, чтобы не
    рисковать переполнением и не ломать ID-колонки (``user_id`` и т.п.).

    Parameters
    ----------
    lf : polars.LazyFrame
        Входной LazyFrame с произвольными числовыми типами.

    Returns
    -------
    polars.LazyFrame
        LazyFrame с Float64-колонками, приведенными к Float32.
        Если Float64-колонок нет, возвращается исходный ``lf`` без изменений.
    """
    schema = lf.collect_schema()
    casts = [pl.col(c).cast(pl.Float32) for c, dtype in schema.items() if dtype == pl.Float64]
    return lf.with_columns(casts) if casts else lf


def _target_columns(target_type: TargetType) -> list[str]:
    """Имена колонок таргета, которые создает :func:`create_target`.

    Parameters
    ----------
    target_type : TargetType
        Тип таргета — один из значений :data:`TargetType`.

    Returns
    -------
    list[str]
        Для ``"multiclass"`` — список из четырех колонок
        ``["target_view", "target_clickout", "target_like", "target_click"]``.
        Для всех остальных типов — ``["target"]``.
    """
    if target_type == "multiclass":
        return list(_MULTICLASS_TARGET_COLS)
    return ["target"]


def _action_target_mapping(raw_files: list[Path], target_type: TargetType) -> pl.DataFrame:
    """Считает стабильный маппинг ``action_type -> target`` через :func:`create_target`.

    ``create_target`` выводит таргет из глобальных счетчиков событий по
    ``action_type``. Чтобы маппинг не зависел от дневных счетчиков (и был
    одинаковым для всех дней), считаем его один раз по всему raw-слою и
    возвращаем дедуплицированную таблицу-справочник.

    Parameters
    ----------
    raw_files : list[Path]
        Список путей к parquet-файлам raw-слоя, содержащих колонку
        ``action_type``.
    target_type : TargetType
        Тип таргета — один из значений :data:`TargetType`.

    Returns
    -------
    polars.DataFrame
        DataFrame с колонками ``["action_type", *target_cols]``, где каждый
        ``action_type`` встречается ровно один раз.
    """
    target_cols = _target_columns(target_type)
    events_lf = pl.scan_parquet(raw_files).select("action_type")
    mapping_lf = create_target(events_lf, target_type=target_type)
    return mapping_lf.select(["action_type", *target_cols]).unique().collect(engine="streaming")


# ---------------------------------------------------------------------------
# Шаг 0. Выборка топовых пользователей и айтемов
# ---------------------------------------------------------------------------


def select_top_entities(
    day_from: int | None = DEFAULT_DAY_FROM,
    day_to: int | None = DEFAULT_DAY_TO,
    n_last_days: int = 10,
    top_users: int | None = 20_000,
    top_items: int | None = 20_000,
    events_dir: Path = EVENTS_RAW_DIR,
    users_path: Path = SELECTED_USERS_PATH,
    items_path: Path = SELECTED_ITEMS_PATH,
) -> tuple[Path, Path]:
    """Выбирает наиболее активных пользователей и айтемы за последние ``n_last_days``.

    Если ``day_from`` / ``day_to`` не заданы, диапазон определяется
    автоматически по содержимому ``events_dir``.

    Списки сохраняются в виде однокоронных parquet-таблиц и дальше переиспользуются
    везде как фильтры ``selected_users`` / ``selected_items``.

    Parameters
    ----------
    day_from : int or None
        Начальный день диапазона или ``None`` для автодетекта.
    day_to : int or None
        Конечный день диапазона или ``None`` для автодетекта.
    n_last_days : int
        Число последних дней диапазона, по которым строится рейтинг
        активности.
    top_users : int or None
        Сколько топовых пользователей сохранить. Если ``None`` —
        сохраняются все уникальные пользователи без ограничения.
    top_items : int or None
        Сколько топовых айтемов сохранить. Если ``None`` —
        сохраняются все уникальные айтемы без ограничения.
    events_dir : Path
        Директория с файлами событий marketplace.
    users_path : Path
        Куда сохранить parquet-файл с выбранными пользователями.
    items_path : Path
        Куда сохранить parquet-файл с выбранными айтемами.

    Returns
    -------
    tuple[Path, Path]
        Кортеж ``(users_path, items_path)`` — пути к созданным файлам.

    Raises
    ------
    FileNotFoundError
        Если в ``events_dir`` не найдено файлов событий за указанный период.
    """
    day_from, day_to = _resolve_day_range(day_from, day_to, events_dir)
    sample_days = list(range(max(day_from, day_to - n_last_days + 1), day_to + 1))
    files = _existing_event_files(sample_days, events_dir)
    if not files:
        raise FileNotFoundError(
            f"В {events_dir} не найдено файлов событий marketplace "
            f"для дней [{sample_days[0]}, {sample_days[-1]}]."
        )

    logger.info(
        "Выбираем {n_users} пользователей / {n_items} айтемов за {n_days} дн...",
        n_users=f"топ-{top_users}" if top_users is not None else "всех",
        n_items=f"топ-{top_items}" if top_items is not None else "всех",
        n_days=len(files),
    )

    events = pl.scan_parquet(files)

    users_lf = events.group_by("user_id").agg(pl.len().alias("len")).sort("len", descending=True)
    if top_users is not None:
        users_lf = users_lf.head(top_users)
    selected_users = users_lf.select("user_id").collect(engine="streaming")

    items_lf = events.group_by("item_id").agg(pl.len().alias("len")).sort("len", descending=True)
    if top_items is not None:
        items_lf = items_lf.head(top_items)
    selected_items = items_lf.select("item_id").collect(engine="streaming")

    users_path.parent.mkdir(parents=True, exist_ok=True)
    items_path.parent.mkdir(parents=True, exist_ok=True)
    selected_users.write_parquet(users_path)
    selected_items.write_parquet(items_path)
    logger.success(
        "Сохранено {n_users} пользователей в {users_path} и {n_items} айтемов в {items_path}.",
        n_users=selected_users.height,
        users_path=users_path,
        n_items=selected_items.height,
        items_path=items_path,
    )
    return users_path, items_path


# ---------------------------------------------------------------------------
# Шаг 1. Raw-слой: разделение событий по action_type и дню
# ---------------------------------------------------------------------------


def build_raw_layer(
    day_from: int | None = DEFAULT_DAY_FROM,
    day_to: int | None = DEFAULT_DAY_TO,
    events_dir: Path = EVENTS_RAW_DIR,
    output_dir: Path = RAW_LAYER_DIR,
    selected_users_path: Path = SELECTED_USERS_PATH,
    selected_items_path: Path = SELECTED_ITEMS_PATH,
    action_types: list[str] | None = None,
) -> Path:
    """Материализует raw-слой ``{output_dir}/{action}/{day}.pq``.

    Каждый файл событий marketplace уже соответствует ровно одному дню (колонка
    ``day``), поэтому идем по файлам по дням и режем их по ``action_type``.
    Пользователи и айтемы фильтруются по предварительно отобранному подмножеству.

    Если ``day_from`` / ``day_to`` не заданы, диапазон определяется
    автоматически по содержимому ``events_dir``.

    Parameters
    ----------
    day_from : int or None
        Начальный день диапазона или ``None`` для автодетекта.
    day_to : int or None
        Конечный день диапазона или ``None`` для автодетекта.
    events_dir : Path
        Директория с исходными файлами событий marketplace.
    output_dir : Path
        Корневая директория raw-слоя datamart; подкаталоги
        ``{action_type}/`` создаются автоматически.
    selected_users_path : Path
        Путь к parquet с отобранными ``user_id``.
    selected_items_path : Path
        Путь к parquet с отобранными ``item_id``.
    action_types : list[str] or None
        Список типов действий для фильтрации; если ``None``,
        используется :data:`ACTION_TYPES`.

    Returns
    -------
    Path
        Путь к корневой директории ``output_dir`` raw-слоя.

    Raises
    ------
    FileNotFoundError
        Если в ``events_dir`` нет файлов для указанного диапазона дней.
    """
    actions = action_types or ACTION_TYPES
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_users = _read_id_list(selected_users_path, "user_id")
    selected_items = _read_id_list(selected_items_path, "item_id")

    day_from, day_to = _resolve_day_range(day_from, day_to, events_dir)
    days = list(range(day_from, day_to + 1))
    files = _existing_event_files(days, events_dir)
    if not files:
        raise FileNotFoundError(
            f"В {events_dir} не найдено файлов событий marketplace для дней "
            f"[{day_from}, {day_to}]."
        )

    logger.info("Сборка raw-слоя в {dir} за {n} дн...", dir=output_dir, n=len(files))
    for day in tqdm(days, desc="Raw-слой (по дням)"):
        src = _event_file(day, events_dir)
        if not src.exists():
            continue
        day_lf = pl.scan_parquet(src).filter(
            pl.col("user_id").is_in(selected_users),
            pl.col("item_id").is_in(selected_items),
        )
        for action in actions:
            out_path = output_dir / action / f"{day}.pq"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            day_lf.filter(pl.col("action_type") == action).sink_parquet(
                out_path, engine="streaming"
            )
    logger.success("Raw-слой собран в {dir}", dir=output_dir)
    return output_dir


# ---------------------------------------------------------------------------
# Шаг 2. Слой агрегаций: дневные счетчики по парам (user, item)
# ---------------------------------------------------------------------------


def build_agg_layer(
    day_from: int | None = DEFAULT_DAY_FROM,
    day_to: int | None = DEFAULT_DAY_TO,
    events_dir: Path = EVENTS_RAW_DIR,
    output_dir: Path = AGG_LAYER_DIR,
    selected_users_path: Path = SELECTED_USERS_PATH,
    selected_items_path: Path = SELECTED_ITEMS_PATH,
) -> Path:
    """Материализует дневные агрегированные счетчики ``{output_dir}/{day}.pq``.

    Для каждой пары (``user_id``, ``item_id``) считаем число событий для каждой
    комбинации ``action_type`` x ``subdomain`` плюс агрегат ``all_subdomains``,
    после чего сводим все pivot-ом в широкую таблицу. Строки с ``null``-значением
    ``subdomain`` (неизвестное поддомен) отбрасываются.

    Если ``day_from`` / ``day_to`` не заданы, диапазон определяется
    автоматически по содержимому ``events_dir``.

    Parameters
    ----------
    day_from : int or None
        Начальный день диапазона или ``None`` для автодетекта.
    day_to : int or None
        Конечный день диапазона или ``None`` для автодетекта.
    events_dir : Path
        Директория с исходными файлами событий marketplace.
    output_dir : Path
        Директория для сохранения агрегаций (``{day}.pq``).
    selected_users_path : Path
        Путь к parquet с отобранными ``user_id``.
    selected_items_path : Path
        Путь к parquet с отобранными ``item_id``.

    Returns
    -------
    Path
        Путь к директории ``output_dir`` со слоем агрегаций.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_users = _read_id_list(selected_users_path, "user_id")
    selected_items = _read_id_list(selected_items_path, "item_id")

    day_from, day_to = _resolve_day_range(day_from, day_to, events_dir)
    days = list(range(day_from, day_to + 1))
    logger.info("Сборка слоя агрегаций в {dir} за {n} дн...", dir=output_dir, n=len(days))

    for day in tqdm(days, desc="Слой агрегаций (по дням)"):
        src = _event_file(day, events_dir)
        if not src.exists():
            continue

        events = pl.scan_parquet(src).filter(
            pl.col("user_id").is_in(selected_users),
            pl.col("item_id").is_in(selected_items),
        )

        by_subdomains_lf = (
            events.group_by("user_id", "item_id", "action_type", "subdomain")
            .agg(pl.len().alias("count"))
            .with_columns(
                ("num_" + pl.col("action_type") + "_" + pl.col("subdomain")).alias("feature")
            )
            .select("user_id", "item_id", "feature", "count")
        )
        all_subdomains_lf = (
            events.group_by("user_id", "item_id", "action_type")
            .agg(pl.len().alias("count"))
            .with_columns(("num_" + pl.col("action_type") + "_all_subdomains").alias("feature"))
            .select("user_id", "item_id", "feature", "count")
        )

        result = (
            pl.concat(
                [
                    by_subdomains_lf.collect(engine="streaming"),
                    all_subdomains_lf.collect(engine="streaming"),
                ]
            )
            # Строки с subdomain=null дают feature=null — их выбрасываем.
            .drop_nulls("feature")
            .pivot(on="feature", index=["user_id", "item_id"], values="count")
            .fill_null(0)
        )
        result.write_parquet(output_dir / f"{day}.pq")

    logger.success("Слой агрегаций собран в {dir}", dir=output_dir)
    return output_dir


# ---------------------------------------------------------------------------
# Шаг 3. Слой признаков: признаки на скользящем окне по группировкам сущностей
# ---------------------------------------------------------------------------


def calculate_features(
    user_group: UserGroup | None,
    item_group: ItemGroup | None,
    day_from: int | None = DEFAULT_DAY_FROM,
    day_to: int | None = DEFAULT_DAY_TO,
    *,
    num_days: int = 30,
    agg_dir: Path = AGG_LAYER_DIR,
    feature_dir: Path = FEATURE_LAYER_DIR,
    users_path: Path = USERS_RAW_PATH,
    items_path: Path = ITEMS_RAW_PATH,
) -> Path:
    """Считает признаки на скользящем окне для заданной группировки сущностей.

    Группировка задается парой ``(user_group, item_group)`` — хотя бы одна из
    них должна быть указана. Примеры:

    * ``("user_id", None)`` → признаки по пользователю;
    * ``(None, "item_id")`` → признаки по айтему;
    * ``("user_id", "item_id")`` → по паре (user, item);
    * ``("socdem_cluster", "category")`` → по (socdem, category).

    Для каждого дня из ``[day_from, day_to]`` агрегируем счетчики из слоя
    агрегаций за окно из предыдущих ``num_days`` дней плюс статистики цены
    по айтему и сохраняем таблицу в ``{feature_dir}/{group_key}/{day}.pq``.

    Если ``day_from`` / ``day_to`` не заданы, диапазон определяется
    автоматически по содержимому ``agg_dir``.

    Parameters
    ----------
    user_group : {"user_id", "region", "socdem_cluster"} or None
        Ключ группировки по пользователю или ``None``.
    item_group : {"item_id", "brand_id", "category", "subcategory"} or None
        Ключ группировки по айтему или ``None``.
    day_from : int or None
        Начальный день диапазона или ``None`` для автодетекта.
    day_to : int or None
        Конечный день диапазона или ``None`` для автодетекта.
    num_days : int
        Размер скользящего окна в днях.
    agg_dir : Path
        Директория со слоем агрегаций (``{day}.pq``).
    feature_dir : Path
        Корневая директория слоя признаков; подкаталог ``{group_key}/``
        создается автоматически.
    users_path : Path
        Путь к parquet с атрибутами пользователей.
    items_path : Path
        Путь к parquet с атрибутами айтемов (должен содержать колонку
        ``price``).

    Returns
    -------
    Path
        Путь к директории ``{feature_dir}/{group_key}/`` с признаками.

    Raises
    ------
    ValueError
        Если оба аргумента ``user_group`` и ``item_group`` равны ``None``.
    """
    if user_group is None and item_group is None:
        raise ValueError("Должен быть задан хотя бы один из user_group / item_group.")

    day_from, day_to = _resolve_day_range(day_from, day_to, agg_dir)

    group_keys: list[str] = [k for k in (user_group, item_group) if k is not None]
    out_dir = feature_dir / "-".join(group_keys)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = list(
        pl.scan_parquet(agg_dir).drop("user_id", "item_id").collect_schema().keys()
    )
    cols_mean_expr = {f"mean_{c}": pl.col(c).mean() for c in feature_cols}
    cols_sum_expr = {f"sum_{c}": pl.col(c).sum() for c in feature_cols}
    price_expr = {
        "mean_price": pl.col("price").mean(),
        "median_price": pl.col("price").median(),
        "min_price": pl.col("price").min(),
        "max_price": pl.col("price").max(),
    }

    needs_user_attrs = user_group not in (None, "user_id")
    users_lf = pl.scan_parquet(users_path) if needs_user_attrs else None

    item_cols = ["item_id", "price"]
    if item_group not in (None, "item_id") and item_group is not None:
        item_cols.append(item_group)
    items_lf = pl.scan_parquet(items_path).select(item_cols)

    desc = f"Признаки {'-'.join(group_keys)} (за {num_days}д)"
    for day in tqdm(range(day_from, day_to + 1), desc=desc):
        window_start = max(day_from, day - num_days + 1)
        window_files = [
            agg_dir / f"{w}.pq"
            for w in range(window_start, day + 1)
            if (agg_dir / f"{w}.pq").exists()
        ]
        if not window_files:
            continue

        agg_window = pl.concat([pl.scan_parquet(p) for p in window_files], how="diagonal_relaxed")

        if needs_user_attrs and users_lf is not None:
            agg_window = agg_window.join(
                users_lf.select(["user_id", user_group]),  # type: ignore[list-item]
                on="user_id",
                how="left",
            )
        agg_window = agg_window.join(items_lf, on="item_id", how="left")

        mean_aggs = agg_window.group_by(group_keys).agg(
            [
                expr.alias(_feature_name(name, group_keys, num_days=num_days))
                for name, expr in cols_mean_expr.items()
            ]
            + [
                expr.alias(_feature_name(name, group_keys, num_days=num_days))
                for name, expr in price_expr.items()
            ]
        )
        sum_aggs = agg_window.group_by(group_keys).agg(
            [
                expr.alias(_feature_name(name, group_keys, num_days=num_days))
                for name, expr in cols_sum_expr.items()
            ]
        )

        day_features = mean_aggs.join(sum_aggs, on=group_keys, how="inner")
        _compact_dtypes(day_features).sink_parquet(out_dir / f"{day}.pq", engine="streaming")

    logger.success("Признаки для {keys} сохранены в {dir}", keys=group_keys, dir=out_dir)
    return out_dir


def build_feature_layer(
    day_from: int | None = DEFAULT_DAY_FROM,
    day_to: int | None = DEFAULT_DAY_TO,
    num_days: int = 30,
    feature_groups: list[tuple[UserGroup | None, ItemGroup | None]] | None = None,
    agg_dir: Path = AGG_LAYER_DIR,
    feature_dir: Path = FEATURE_LAYER_DIR,
    users_path: Path = USERS_RAW_PATH,
    items_path: Path = ITEMS_RAW_PATH,
) -> Path:
    """Запускает :func:`calculate_features` для всех заданных группировок.

    Если ``day_from`` / ``day_to`` не заданы, диапазон определяется
    автоматически по содержимому ``agg_dir``.

    Parameters
    ----------
    day_from : int or None
        Начальный день диапазона или ``None`` для автодетекта.
    day_to : int or None
        Конечный день диапазона или ``None`` для автодетекта.
    num_days : int
        Размер скользящего окна в днях для :func:`calculate_features`.
    feature_groups : list[tuple] or None
        Список пар ``(user_group, item_group)`` для обработки.
        Если ``None``, используется :data:`DEFAULT_FEATURE_GROUPS`.
    agg_dir : Path
        Директория со слоем агрегаций.
    feature_dir : Path
        Корневая директория слоя признаков.
    users_path : Path
        Путь к parquet с атрибутами пользователей.
    items_path : Path
        Путь к parquet с атрибутами айтемов.

    Returns
    -------
    Path
        Путь к корневой директории ``feature_dir`` слоя признаков.
    """
    day_from, day_to = _resolve_day_range(day_from, day_to, agg_dir)
    groups = feature_groups or DEFAULT_FEATURE_GROUPS
    logger.info("Сборка слоя признаков для {n} группировок...", n=len(groups))
    for user_group, item_group in groups:
        calculate_features(
            user_group=user_group,
            item_group=item_group,
            day_from=day_from,
            day_to=day_to,
            num_days=num_days,
            agg_dir=agg_dir,
            feature_dir=feature_dir,
            users_path=users_path,
            items_path=items_path,
        )
    logger.success("Слой признаков собран в {dir}", dir=feature_dir)
    return feature_dir


# ---------------------------------------------------------------------------
# Шаг 4. Слой basis: таблицы (session_id, user_id, item_id, label)
# ---------------------------------------------------------------------------


def build_basis(
    day_from: int | None = DEFAULT_DAY_FROM,
    day_to: int | None = DEFAULT_DAY_TO,
    raw_dir: Path = RAW_LAYER_DIR,
    output_dir: Path = BASIS_DIR,
    filter_99: bool = True,
    drop_view_only_sessions: bool = True,
    target_type: TargetType = DEFAULT_TARGET_TYPE,
    action_types: list[str] | None = None,
) -> Path:
    """Строит дневные таблицы ранжировочной основы (basis).

    Строка имеет вид ``(session_id, day, user_id, item_id, <колонки таргета>)``,
    где ``session_id`` = ``f"{day}__{user_id}"``, ``day`` — целочисленный
    номер дня (``Int32``, совместим с :func:`src.dataset.global_temporal_split`),
    а колонки таргета формирует :func:`src.dataset.create_target`:

    * ``log_target`` / ``sqrt_target`` / ``unproccessed`` — одна скалярная
      колонка ``label`` (``target`` переименован для удобства);
    * ``multiclass`` — четыре бинарные колонки ``target_view``,
      ``target_clickout``, ``target_like``, ``target_click``.

    Маппинг ``action_type -> target`` считается **один раз** по всем дням
    из ``[day_from, day_to]``, чтобы одному и тому же типу действия всегда
    ставилось одно и то же численное значение независимо от дня.

    Сессии, состоящие только из событий ``view``, выбрасываются: их нельзя
    ранжировать (у всех айтемов одинаковый label). При ``filter_99=True``
    также удаляются сессии длиннее 99-го перцентиля длины.

    Если ``day_from`` / ``day_to`` не заданы, диапазон определяется
    автоматически по содержимому ``raw_dir`` (структура ``{action}/{day}.pq``).

    Parameters
    ----------
    day_from : int or None
        Начальный день диапазона или ``None`` для автодетекта.
    day_to : int or None
        Конечный день диапазона или ``None`` для автодетекта.
    raw_dir : Path
        Корневая директория raw-слоя (``{action_type}/{day}.pq``).
    output_dir : Path
        Директория для сохранения basis-файлов (``{day}.pq``).
    filter_99 : bool
        Если ``True``, удалять сессии длиннее 99-го перцентиля длины сессии.
    drop_view_only_sessions : bool
        Если ``True``, удалять сессии, в которых все события имеют
        ``action_type == "view"``.
    target_type : TargetType
        Способ расчета таргета; см. :data:`TargetType`.
    action_types : list[str] or None
        Список типов действий для обработки; если ``None``,
        используется :data:`ACTION_TYPES`.

    Returns
    -------
    Path
        Путь к директории ``output_dir`` со слоем basis.

    Raises
    ------
    FileNotFoundError
        Если в ``raw_dir`` не найдено файлов raw-слоя для указанного
        диапазона дней.
    """
    actions = action_types or ACTION_TYPES
    output_dir.mkdir(parents=True, exist_ok=True)

    day_from, day_to = _resolve_day_range(day_from, day_to, raw_dir, "*/*.pq")
    days = list(range(day_from, day_to + 1))
    raw_files = [
        raw_dir / action / f"{day}.pq"
        for action in actions
        for day in days
        if (raw_dir / action / f"{day}.pq").exists()
    ]
    if not raw_files:
        raise FileNotFoundError(
            f"В {raw_dir} нет файлов raw-слоя для дней [{day_from}, {day_to}]."
        )

    target_cols = _target_columns(target_type)
    label_cols = target_cols if target_type == "multiclass" else ["label"]

    logger.info(
        "Сборка basis (target_type={t}) из {raw} в {out} за дни [{a}, {b}]",
        t=target_type,
        raw=raw_dir,
        out=output_dir,
        a=day_from,
        b=day_to,
    )

    target_mapping_df = _action_target_mapping(raw_files, target_type)
    target_mapping_lf = target_mapping_df.lazy()

    for day in tqdm(days, desc="Basis (по дням)"):
        day_files = [
            raw_dir / action / f"{day}.pq"
            for action in actions
            if (raw_dir / action / f"{day}.pq").exists()
        ]
        if not day_files:
            continue

        day_events = (
            pl.scan_parquet(day_files)
            .select("user_id", "item_id", "action_type")
            .join(target_mapping_lf, on="action_type")
            .with_columns(
                (pl.lit(f"{day}__") + pl.col("user_id").cast(pl.String)).alias("session_id")
            )
        )

        basis_day = day_events.group_by("session_id", "user_id", "item_id").agg(
            [pl.col(c).max() for c in target_cols]
        )
        if target_type != "multiclass":
            basis_day = basis_day.rename({"target": "label"})

        sessions = day_events.group_by("session_id").agg(pl.len().alias("len"))
        if drop_view_only_sessions:
            non_view_sessions = (
                day_events.filter(pl.col("action_type") != "view").select("session_id").unique()
            )
            sessions = sessions.join(non_view_sessions, on="session_id", how="semi")
        if filter_99:
            len_p99 = sessions.select(pl.col("len").quantile(0.99, "nearest")).collect().item()
            if len_p99 is not None:
                sessions = sessions.filter(pl.col("len") <= len_p99)

        basis_lf = (
            basis_day.join(sessions.select("session_id"), on="session_id", how="inner")
            .with_columns(pl.lit(day, dtype=pl.Int32).alias("day"))
            .select(["session_id", "day", "user_id", "item_id", *label_cols])
        )
        _compact_dtypes(basis_lf).sink_parquet(output_dir / f"{day}.pq", engine="streaming")

    logger.success("Basis (target_type={t}) собран в {dir}", t=target_type, dir=output_dir)
    return output_dir


# ---------------------------------------------------------------------------
# Шаг 5. Слой dataset: basis + признаки + атрибуты user/item
# ---------------------------------------------------------------------------


def build_dataset(
    day_from: int | None = None,
    day_to: int | None = DEFAULT_DAY_TO,
    basis_dir: Path = BASIS_DIR,
    feature_dir: Path = FEATURE_LAYER_DIR,
    output_dir: Path = DATASET_DIR,
    features_to_use_filepath: Path | None = None,
    users_path: Path = USERS_RAW_PATH,
    items_path: Path = ITEMS_RAW_PATH,
    selected_users_path: Path = SELECTED_USERS_PATH,
    selected_items_path: Path = SELECTED_ITEMS_PATH,
) -> Path:
    """Склеивает basis с признаками предыдущего дня и атрибутами сущностей.

    Чтобы избежать временной утечки, признаки для дня ``D`` берутся из
    ``{feature_dir}/.../{D-1}.pq``. Колонки-атрибуты user и item
    переименовываются в ``f_num__*`` / ``f_cat__*``, чтобы ранкер мог
    подхватывать их по префиксу.

    Если ``day_from`` / ``day_to`` не заданы, диапазон определяется
    автоматически по содержимому ``basis_dir``: ``day_from`` = минимальный
    день basis-а + 1 (нужны признаки за предыдущий день), ``day_to`` =
    максимальный день basis-а.

    Parameters
    ----------
    day_from : int or None
        Начальный день диапазона или ``None`` для автодетекта
        (``min_basis_day + 1``).
    day_to : int or None
        Конечный день диапазона или ``None`` для автодетекта
        (``max_basis_day``).
    basis_dir : Path
        Директория со слоем basis (``{day}.pq``).
    feature_dir : Path
        Корневая директория слоя признаков с подкаталогами по группировкам.
    output_dir : Path
        Директория для сохранения финальных датасетов (``{day}.pq``).
    features_to_use_filepath : Path or None
        Путь к файлу со списком групп признаков для включения (JSON или
        plain-text). Если ``None``, берутся все подкаталоги ``feature_dir``.
    users_path : Path
        Путь к parquet с атрибутами пользователей.
    items_path : Path
        Путь к parquet с атрибутами айтемов (колонка ``embedding`` будет
        удалена при наличии).
    selected_users_path : Path
        Путь к parquet с отобранными ``user_id``.
    selected_items_path : Path
        Путь к parquet с отобранными ``item_id``.

    Returns
    -------
    Path
        Путь к директории ``output_dir`` с итоговым датасетом.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    auto_from, auto_to = _discover_day_range(basis_dir)
    if day_from is None:
        day_from = auto_from + 1
    if day_to is None:
        day_to = auto_to

    selected_users_lf = pl.scan_parquet(selected_users_path)
    selected_items_lf = pl.scan_parquet(selected_items_path)

    users_lf: pl.LazyFrame | None = None
    if users_path.exists():
        users_lf = pl.scan_parquet(users_path).join(selected_users_lf, on="user_id", how="semi")
    items_lf: pl.LazyFrame | None = None
    if items_path.exists():
        items_lf = (
            pl.scan_parquet(items_path)
            .drop("embedding", strict=False)
            .join(selected_items_lf, on="item_id", how="semi")
        )

    user_rename, user_cat_cols = (
        _attribute_rename_map(users_lf, "user_id") if users_lf is not None else ({}, [])
    )
    item_rename, item_cat_cols = (
        _attribute_rename_map(items_lf, "item_id") if items_lf is not None else ({}, [])
    )
    rename_map = {**user_rename, **item_rename}
    cat_cols = user_cat_cols + item_cat_cols

    features_to_use = _load_features_to_use(features_to_use_filepath)
    if features_to_use is None:
        feature_folders = sorted(p.name for p in feature_dir.iterdir() if p.is_dir())
    else:
        feature_folders = features_to_use
    logger.info("Сборка датасета на {n} группах признаков", n=len(feature_folders))

    for day in tqdm(range(day_from, day_to + 1), desc="Dataset (по дням)"):
        basis_file = basis_dir / f"{day}.pq"
        if not basis_file.exists():
            continue

        df = pl.scan_parquet(basis_file)
        if users_lf is not None:
            df = df.join(users_lf, on="user_id", how="left")
        if items_lf is not None:
            df = df.join(items_lf, on="item_id", how="left")

        prev_day_file = f"{day - 1}.pq"
        for folder in feature_folders:
            feature_file = feature_dir / folder / prev_day_file
            if not feature_file.exists():
                continue
            join_keys = folder.split("-")
            df = df.join(pl.scan_parquet(feature_file), on=join_keys, how="left")

        if rename_map:
            df = df.rename(rename_map)
        if cat_cols:
            df = df.with_columns(
                [pl.col(c).cast(pl.String).cast(pl.Categorical) for c in cat_cols]
            )

        _compact_dtypes(df).sink_parquet(output_dir / f"{day}.pq", engine="streaming")

    logger.success("Датасет собран в {dir}", dir=output_dir)
    return output_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command("select-entities")
def cli_select_entities(
    day_from: int | None = DEFAULT_DAY_FROM,
    day_to: int | None = DEFAULT_DAY_TO,
    n_last_days: int = 10,
    top_users: int | None = 20_000,
    top_items: int | None = 20_000,
    events_dir: Path = EVENTS_RAW_DIR,
    users_path: Path = SELECTED_USERS_PATH,
    items_path: Path = SELECTED_ITEMS_PATH,
) -> None:
    """Отбирает топ-N самых активных пользователей и айтемов и сохраняет их в processed/.

    Обертка CLI над :func:`select_top_entities`. Результат записывается в
    ``selected_users.pq`` / ``selected_items.pq`` и используется всеми
    последующими шагами как фильтр.

    Для выбора всех пользователей/айтемов передайте ``--top-users 0`` / ``--top-items 0`` (или ``None``).
    """
    select_top_entities(
        day_from=day_from,
        day_to=day_to,
        n_last_days=n_last_days,
        top_users=top_users,
        top_items=top_items,
        events_dir=events_dir,
        users_path=users_path,
        items_path=items_path,
    )


@app.command("build-raw")
def cli_build_raw(
    day_from: int | None = DEFAULT_DAY_FROM,
    day_to: int | None = DEFAULT_DAY_TO,
    events_dir: Path = EVENTS_RAW_DIR,
    output_dir: Path = RAW_LAYER_DIR,
    selected_users_path: Path = SELECTED_USERS_PATH,
    selected_items_path: Path = SELECTED_ITEMS_PATH,
) -> None:
    """Собирает RAW-слой datamart (события, разделенные по action_type x день).

    Обертка CLI над :func:`build_raw_layer`. Читает события marketplace,
    фильтрует по отобранным пользователям/айтемам и записывает разбитые
    по ``action_type`` parquet-файлы в ``datamart/raw/events/``.
    """
    build_raw_layer(
        day_from=day_from,
        day_to=day_to,
        events_dir=events_dir,
        output_dir=output_dir,
        selected_users_path=selected_users_path,
        selected_items_path=selected_items_path,
    )


@app.command("build-aggs")
def cli_build_aggs(
    day_from: int | None = DEFAULT_DAY_FROM,
    day_to: int | None = DEFAULT_DAY_TO,
    events_dir: Path = EVENTS_RAW_DIR,
    output_dir: Path = AGG_LAYER_DIR,
    selected_users_path: Path = SELECTED_USERS_PATH,
    selected_items_path: Path = SELECTED_ITEMS_PATH,
) -> None:
    """Собирает слой АГРЕГАЦИЙ datamart (дневные счетчики по парам (user, item)).

    Обертка CLI над :func:`build_agg_layer`. Для каждого дня считает pivot-таблицу
    событий по ``action_type`` x ``subdomain`` и записывает результат в
    ``datamart/aggs/events/``.
    """
    build_agg_layer(
        day_from=day_from,
        day_to=day_to,
        events_dir=events_dir,
        output_dir=output_dir,
        selected_users_path=selected_users_path,
        selected_items_path=selected_items_path,
    )


@app.command("build-features")
def cli_build_features(
    day_from: int | None = DEFAULT_DAY_FROM,
    day_to: int | None = DEFAULT_DAY_TO,
    num_days: int = 30,
    agg_dir: Path = AGG_LAYER_DIR,
    feature_dir: Path = FEATURE_LAYER_DIR,
    users_path: Path = USERS_RAW_PATH,
    items_path: Path = ITEMS_RAW_PATH,
) -> None:
    """Собирает слой ПРИЗНАКОВ datamart (признаки на скользящем окне по группировкам).

    Обертка CLI над :func:`build_feature_layer`. Запускает расчет признаков для
    всех группировок из :data:`DEFAULT_FEATURE_GROUPS` и записывает результат в
    ``datamart/features/events/``.
    """
    build_feature_layer(
        day_from=day_from,
        day_to=day_to,
        num_days=num_days,
        agg_dir=agg_dir,
        feature_dir=feature_dir,
        users_path=users_path,
        items_path=items_path,
    )


@app.command("build-basis")
def cli_build_basis(
    day_from: int | None = DEFAULT_DAY_FROM,
    day_to: int | None = DEFAULT_DAY_TO,
    raw_dir: Path = RAW_LAYER_DIR,
    output_dir: Path = BASIS_DIR,
    filter_99: bool = True,
    drop_view_only_sessions: bool = True,
    target_type: str = typer.Option(
        DEFAULT_TARGET_TYPE,
        help=(
            "Способ расчета таргета через src.dataset.create_target: "
            "log_target, sqrt_target, unproccessed или multiclass."
        ),
    ),
) -> None:
    """Собирает ранжировочный BASIS (session_id, user_id, item_id, колонки таргета).

    Обертка CLI над :func:`build_basis`. Читает raw-слой, агрегирует события
    по сессиям, вычисляет таргет через :func:`src.dataset.create_target` и
    записывает basis-файлы в ``processed/basis/``.
    """
    build_basis(
        day_from=day_from,
        day_to=day_to,
        raw_dir=raw_dir,
        output_dir=output_dir,
        filter_99=filter_99,
        drop_view_only_sessions=drop_view_only_sessions,
        target_type=t.cast(TargetType, target_type),
    )


@app.command("build-dataset")
def cli_build_dataset(
    day_from: int | None = None,
    day_to: int | None = DEFAULT_DAY_TO,
    basis_dir: Path = BASIS_DIR,
    feature_dir: Path = FEATURE_LAYER_DIR,
    output_dir: Path = DATASET_DIR,
    features_to_use_filepath: Path | None = None,
    users_path: Path = USERS_RAW_PATH,
    items_path: Path = ITEMS_RAW_PATH,
    selected_users_path: Path = SELECTED_USERS_PATH,
    selected_items_path: Path = SELECTED_ITEMS_PATH,
) -> None:
    """Собирает обучающий ДАТАСЕТ (basis + признаки за предыдущий день + атрибуты).

    Обертка CLI над :func:`build_dataset`. Присоединяет к basis-таблице признаки
    за предыдущий день и атрибуты пользователей/айтемов; результат записывается
    в ``processed/dataset/``.
    """
    build_dataset(
        day_from=day_from,
        day_to=day_to,
        basis_dir=basis_dir,
        feature_dir=feature_dir,
        output_dir=output_dir,
        features_to_use_filepath=features_to_use_filepath,
        users_path=users_path,
        items_path=items_path,
        selected_users_path=selected_users_path,
        selected_items_path=selected_items_path,
    )


@app.command("all")
def cli_all(
    day_from: int | None = DEFAULT_DAY_FROM,
    day_to: int | None = DEFAULT_DAY_TO,
    num_days: int = 30,
    top_users: int | None = 20_000,
    top_items: int | None = 20_000,
    n_last_days: int = 10,
    skip_select: bool = False,
    target_type: str = typer.Option(
        DEFAULT_TARGET_TYPE,
        help=(
            "Способ расчета таргета через src.dataset.create_target: "
            "log_target, sqrt_target, unproccessed или multiclass."
        ),
    ),
) -> None:
    """Запускает полный пайплайн feature engineering от начала до конца.

    Если ``day_from`` / ``day_to`` не заданы, диапазон определяется
    автоматически: на каждом шаге берется весь доступный набор дней
    из соответствующего источника (raw events / agg / basis).
    """
    if not skip_select:
        select_top_entities(
            day_from=day_from,
            day_to=day_to,
            n_last_days=n_last_days,
            top_users=top_users,
            top_items=top_items,
        )
    build_raw_layer(day_from=day_from, day_to=day_to)
    build_agg_layer(day_from=day_from, day_to=day_to)
    build_feature_layer(day_from=day_from, day_to=day_to, num_days=num_days)
    build_basis(day_from=day_from, day_to=day_to, target_type=t.cast(TargetType, target_type))
    # Для dataset: если день начала задан явно, нужны фичи за day_from-1, поэтому
    # сдвигаем на +1; иначе оставляем None и build_dataset сам автодетектит.
    dataset_day_from = day_from + 1 if day_from is not None else None
    build_dataset(day_from=dataset_day_from, day_to=day_to)


if __name__ == "__main__":
    app()
