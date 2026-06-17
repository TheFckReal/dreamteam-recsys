from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import time
from typing import Iterator

import datasets
from loguru import logger
import numpy as np
import polars as pl
from scipy.sparse import csr_matrix  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402
import typer  # noqa: E402

from src.config import PROCESSED_DATA_DIR, VAE_DIR
from src.modeling.vae import MultiVAERecommender, multivae_loss

app = typer.Typer(help="OOM-обучение Mult-VAE на агрегированных фичах user-item.")


# ---------------------------------------------------------------------------
# Пути и константы
# ---------------------------------------------------------------------------

USER_ITEM_FEATURE_DIR = PROCESSED_DATA_DIR / "datamart" / "features" / "events" / "user_id-item_id"
SELECTED_USERS_PATH = PROCESSED_DATA_DIR / "selected_users.pq"
SELECTED_ITEMS_PATH = PROCESSED_DATA_DIR / "selected_items.pq"

# Веса разных событий.
# Применяются только при ``--no-binary`` (log-confidence режим).
ACTION_WEIGHTS: dict[str, float] = {
    "view": 1.0,
    "click": 2.0,
    "clickout": 4.0,
    "like": 8.0,
}

_FEATURE_TPL = "f_num__user_id_item_id__sum_num_{action}_all_subdomains_30d"


# ---------------------------------------------------------------------------
# Шаг 1. Подготовка sparse user-history (OOM через polars streaming)
# ---------------------------------------------------------------------------


def build_user_history_dataset(
    day: int,
    output_path: Path,
    feature_dir: Path = USER_ITEM_FEATURE_DIR,
    selected_users_path: Path | None = SELECTED_USERS_PATH,
    selected_items_path: Path | None = SELECTED_ITEMS_PATH,
    binary: bool = True,
    min_user_items: int = 1,
) -> tuple[Path, dict[str, int], dict[int, int]]:
    """Готовит sparse user-history parquet для Mult-VAE.

    Использует ``pl.scan_parquet`` + ``sink_parquet``, чтобы не материализовать
    dense user×item матрицу. На выходе — parquet со схемой
    ``(user_idx: u32, item_idx: list[u32], weight: list[f32])``.

    .. warning::
        Feature-снимок ``user_id-item_id/{day}.pq`` агрегирует 30-дневное окно
        ``[day-29, day]`` *включительно*, т.е. знает события самого ``day``.
        Чтобы избежать target-leak, тестовые метрики необходимо считать
        ТОЛЬКО на событиях за дни ``> day``. Никогда не передавайте сюда день,
        попадающий в тестовый период.

    Parameters
    ----------
    day : int
        День, чьи 30-дневные агрегаты будут использованы как снимок history.
    output_path : Path
        Путь для итогового parquet.
    feature_dir : Path
        Папка с дневными feature-файлами по парам ``(user_id, item_id)``.
    selected_users_path, selected_items_path : Path or None
        Канонические наборы пользователей и айтемов. Если ``None`` — берем
        всех пользователей/айтемов, встретившихся в фичах.
    binary : bool
        ``True`` — бинаризация (любое взаимодействие → ``1``). Так делает
        оригинальный Mult-VAE. ``False`` — взвешенная сумма событий с
        ``log1p`` для сжатия хвостов.
    min_user_items : int
        Фильтр: пользователи с менее чем ``min_user_items`` items
        выкидываются (для VAE рекомендуется ≥ 5).

    Returns
    -------
    tuple[Path, dict[str, int], dict[int, int]]
        ``(output_path, item_to_idx, user_to_idx)``.
    """
    feature_path = feature_dir / f"{day}.pq"
    if not feature_path.exists():
        raise FileNotFoundError(f"Нет feature-файла для дня {day}: {feature_path}")

    logger.info("Сбор маппингов item/user...")

    # --- item_to_idx -------------------------------------------------------
    if selected_items_path is not None and Path(selected_items_path).exists():
        items_lf = pl.scan_parquet(selected_items_path).select("item_id")
    else:
        items_lf = pl.scan_parquet(feature_path).select("item_id").unique()

    items_df = items_lf.collect(engine="streaming").unique("item_id").sort("item_id")
    item_to_idx: dict[str, int] = {
        iid: idx for idx, iid in enumerate(items_df["item_id"].to_list())
    }
    logger.info("Каталог айтемов: {} штук", len(item_to_idx))

    # --- агрегация user-history (streaming) -------------------------------
    weight_cols = [_FEATURE_TPL.format(action=a) for a in ACTION_WEIGHTS]

    base = pl.scan_parquet(feature_path).select(["user_id", "item_id", *weight_cols])

    if selected_users_path is not None and Path(selected_users_path).exists():
        users_lf = pl.scan_parquet(selected_users_path).select("user_id").unique()
        base = base.join(users_lf, on="user_id", how="inner")

    # join словаря айтемов — оставляет только items из каталога
    items_idx_lf = pl.LazyFrame(
        {"item_id": list(item_to_idx.keys()), "item_idx": list(item_to_idx.values())},
        schema={"item_id": pl.String, "item_idx": pl.UInt32},
    )
    base = base.join(items_idx_lf, on="item_id", how="inner")

    # Защитимся от потенциальных null'ов в счетчиках: feature-слой по контракту
    # их не содержит (там ``fill_null(0)`` после pivot), но дешевле перестраховаться.
    raw_count = sum(
        (
            pl.col(_FEATURE_TPL.format(action=a)).cast(pl.Float32).fill_null(0.0) * w
            for a, w in ACTION_WEIGHTS.items()
        ),
        start=pl.lit(0.0, dtype=pl.Float32),
    )

    if binary:
        weight_expr = (raw_count > 0).cast(pl.Float32)
    else:
        weight_expr = raw_count.log1p().cast(pl.Float32)

    # Финальный фильтр: убираем zero-weight пары после применения схемы.
    pair_lf = (
        base.with_columns(weight=weight_expr)
        .filter(pl.col("weight") > 0)
        .select("user_id", "item_idx", "weight")
    )

    # Группируем по пользователю в list-колонки -> одна строка = один юзер.
    user_lf = (
        pair_lf.group_by("user_id")
        .agg(
            pl.col("item_idx").alias("item_idx"),
            pl.col("weight").alias("weight"),
        )
        .with_columns(num_items=pl.col("item_idx").list.len())
        .filter(pl.col("num_items") >= min_user_items)
        .sort("user_id")
    )

    # Назначаем user_idx последовательно по отсортированному user_id.
    user_lf = user_lf.with_columns(user_idx=pl.int_range(0, pl.len(), dtype=pl.UInt32))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Сохранение sparse user-history в {} (streaming sink)...", output_path)
    user_lf.select("user_idx", "user_id", "item_idx", "weight").sink_parquet(output_path)

    user_meta = pl.read_parquet(output_path, columns=["user_idx", "user_id"])
    user_to_idx: dict[int, int] = dict(
        zip(user_meta["user_id"].to_list(), user_meta["user_idx"].to_list())
    )
    logger.info("Активных пользователей: {}", len(user_to_idx))

    return output_path, item_to_idx, user_to_idx


# ---------------------------------------------------------------------------
# Шаг 2. OOM-датасет на базе HuggingFace ``datasets`` (Arrow memory-mapped)
# ---------------------------------------------------------------------------


class _SparseHistoryCollator:
    """Кастомный коллатор: sparse history → dense ``(B, num_items)``.

    Dense user×item матрица никогда не материализуется целиком — только в
    рамках текущего батча, после step-а GC возвращает память.
    """

    def __init__(self, num_items: int) -> None:
        self.num_items = num_items

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        bsz = len(batch)
        # Заполняем dense батч поэлементно. Делаем это на CPU; в обучении
        # потом ``.to(device, non_blocking=True)``.
        dense = torch.zeros((bsz, self.num_items), dtype=torch.float32)
        user_idxs = torch.empty(bsz, dtype=torch.int64)
        for row, item in enumerate(batch):
            idx = np.asarray(item["item_idx"], dtype=np.int64)
            w = np.asarray(item["weight"], dtype=np.float32)
            dense[row, idx] = torch.from_numpy(w)
            user_idxs[row] = item["user_idx"]
        return {"history": dense, "user_idx": user_idxs}


def make_oom_dataloader(
    history_path: Path,
    num_items: int,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[DataLoader, datasets.Dataset]:
    """Создает DataLoader поверх memory-mapped ``datasets.Dataset``.

    Returns
    -------
    tuple[DataLoader, datasets.Dataset]
        Сам DataLoader и подлежащий ``Dataset`` (полезен для подсчета длины,
        повторного запуска эпох и т.п.).
    """
    ds = datasets.Dataset.from_parquet(str(history_path), keep_in_memory=False)
    # set_format("python") заставляет Arrow возвращать обычные list/int —
    # без копирования всего блока в numpy/torch заранее.
    ds.set_format(type=None, columns=["user_idx", "item_idx", "weight"])

    collator = _SparseHistoryCollator(num_items=num_items)
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
        generator=g,
    )
    return loader, ds


# ---------------------------------------------------------------------------
# Шаг 3. Тренировочный цикл
# ---------------------------------------------------------------------------


class KLAnnealer:
    """
    Шедулер для убывающего β в KL-члене.

    Parameters
    ----------
    beta_max : float
        Целевое значение β после прогрева.
    total_anneal_steps : int
        Сколько ``step()`` нужно для выхода на ``beta_max``. ``<=0`` —
        annealing отключен, β всегда равен ``beta_max``.
    schedule : {"linear", "cosine"}
        ``"linear"`` — оригинал из Mult-VAE статьи,
        ``"cosine"`` — мягкое плечо, β = beta_max * (1 - cos(π · t/T)) / 2.
    """

    def __init__(
        self,
        beta_max: float,
        total_anneal_steps: int,
        schedule: str = "linear",
    ) -> None:
        if schedule not in {"linear", "cosine"}:
            raise ValueError(f"Unknown schedule: {schedule!r}")
        self.beta_max = float(beta_max)
        self.total_anneal_steps = int(total_anneal_steps)
        self.schedule = schedule
        self._step = 0
        self._last_value = self._compute(0)

    @property
    def last_value(self) -> float:
        """Текущий β (после последнего ``step``)."""
        return self._last_value

    def _compute(self, step: int) -> float:
        if self.total_anneal_steps <= 0:
            return self.beta_max
        t = min(1.0, step / self.total_anneal_steps)
        if self.schedule == "linear":
            return self.beta_max * t
        # cosine
        import math

        return self.beta_max * (1.0 - math.cos(math.pi * t)) / 2.0

    def step(self) -> float:
        """Сдвинуть глобальный счетчик и вернуть свежее значение β."""
        self._step += 1
        self._last_value = self._compute(self._step)
        return self._last_value

    def state_dict(self) -> dict:
        return {
            "step": self._step,
            "beta_max": self.beta_max,
            "total_anneal_steps": self.total_anneal_steps,
            "schedule": self.schedule,
        }

    def load_state_dict(self, state: dict) -> None:
        self._step = int(state["step"])
        self.beta_max = float(state["beta_max"])
        self.total_anneal_steps = int(state["total_anneal_steps"])
        self.schedule = str(state["schedule"])
        self._last_value = self._compute(self._step)


def _iter_epoch(loader: DataLoader, device: str) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    for batch in loader:
        yield (
            batch["history"].to(device, non_blocking=True),
            batch["user_idx"].to(device, non_blocking=True),
        )


def train_vae(
    history_path: Path,
    item_to_idx: dict[str, int],
    user_to_idx: dict[int, int],
    artifacts_dir: Path,
    *,
    encoder_dims: list[int] | None = None,
    dropout: float = 0.5,
    beta: float = 0.2,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    total_anneal_steps: int = 200_000,
    kl_schedule: str = "linear",
    use_lr_scheduler: bool = True,
    num_workers: int = 0,
    device: str | None = None,
    seed: int = 42,
) -> Path:
    """Полный тренинг Mult-VAE с OOM-датасетом и KL-annealing.

    Возвращает путь к сохраненному чекпоинту.
    """
    rec = MultiVAERecommender(
        encoder_dims=encoder_dims,
        dropout=dropout,
        beta=beta,
        device=device,
        random_state=seed,
    )
    rec.set_vocab(user_to_idx, item_to_idx)
    model = rec.build_model()
    assert model is not None

    loader, ds = make_oom_dataloader(
        history_path=history_path,
        num_items=len(item_to_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        seed=seed,
    )
    logger.info(
        "Стартую обучение: {} пользователей, {} items, batches per epoch ≈ {}",
        len(ds),
        len(item_to_idx),
        len(loader),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    kl_annealer = KLAnnealer(
        beta_max=beta,
        total_anneal_steps=total_anneal_steps,
        schedule=kl_schedule,
    )

    # Learning rate scheduler
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    if use_lr_scheduler:
        steps_per_epoch = len(loader)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(steps_per_epoch * epochs, 1),
            eta_min=lr / 100.0,
        )

    # --- сборка sparse user_items для последующего top-k ----------------
    # Sparse матрица хранит только взаимодействия
    rows, cols, vals = [], [], []
    for example in tqdm(ds, total=len(ds), desc="Сборка user_items", unit="user"):
        u = example["user_idx"]
        idx = example["item_idx"]
        w = example["weight"]
        rows.extend([u] * len(idx))
        cols.extend(idx)
        vals.extend(w)
    user_items = csr_matrix(
        (np.asarray(vals, dtype=np.float32), (np.asarray(rows), np.asarray(cols))),
        shape=(len(user_to_idx), len(item_to_idx)),
    )
    rec.set_user_items(user_items)

    # --- собственно цикл -------------------------------------------------
    epoch_bar = tqdm(range(1, epochs + 1), desc="Эпохи", unit="epoch", position=0)
    for epoch in epoch_bar:
        model.train()
        ep_loss = ep_nll = ep_kld = 0.0
        n_batches = 0
        t0 = time.time()

        batch_bar = tqdm(
            _iter_epoch(loader, rec.device),
            total=len(loader),
            desc=f"Epoch {epoch:02d}",
            unit="batch",
            leave=False,
            position=1,
        )
        for history, _user_idxs in batch_bar:
            current_beta = kl_annealer.step()

            optimizer.zero_grad(set_to_none=True)
            logits, mu, logvar = model(history)
            loss, nll, kld = multivae_loss(logits, history, mu, logvar, current_beta)
            loss.backward()
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

            loss_v = float(loss.item())
            nll_v = float(nll.item())
            kld_v = float(kld.item())
            ep_loss += loss_v
            ep_nll += nll_v
            ep_kld += kld_v
            n_batches += 1

            # логирование метрик
            batch_bar.set_postfix(
                loss=f"{loss_v:.3f}",
                nll=f"{nll_v:.3f}",
                kld=f"{kld_v:.3f}",
                beta=f"{current_beta:.3f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                avg_loss=f"{ep_loss / n_batches:.3f}",
            )
        batch_bar.close()

        dt = time.time() - t0
        avg_loss = ep_loss / max(n_batches, 1)
        avg_nll = ep_nll / max(n_batches, 1)
        avg_kld = ep_kld / max(n_batches, 1)
        epoch_bar.set_postfix(
            loss=f"{avg_loss:.3f}",
            nll=f"{avg_nll:.3f}",
            kld=f"{avg_kld:.3f}",
            beta=f"{kl_annealer.last_value:.3f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )
        logger.info(
            "Epoch {:02d}: loss={:.4f} nll={:.4f} kld={:.4f} beta={:.4f} lr={:.2e} time={:.1f}s",
            epoch,
            avg_loss,
            avg_nll,
            avg_kld,
            kl_annealer.last_value,
            optimizer.param_groups[0]["lr"],
            dt,
        )
    epoch_bar.close()

    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifacts_dir / "model.pt"
    rec.save(checkpoint_path)

    # Sparse user-history тоже сохраняем — нужна для filter_already_liked_items
    # на инференсе.
    from scipy.sparse import save_npz

    save_npz(artifacts_dir / "user_items.npz", user_items)
    logger.info("user_items.npz сохранена в {}", artifacts_dir)

    return checkpoint_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def prepare(
    day: int = typer.Option(..., help="День, фичи которого использовать как снимок history."),
    output_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "vae_history.pq",
        help="Куда сохранить sparse user-history parquet.",
    ),
    feature_dir: Path = typer.Option(
        USER_ITEM_FEATURE_DIR, help="Папка с фичами по парам (user_id, item_id)."
    ),
    binary: bool = typer.Option(True, help="Бинаризовать веса (как в оригинальном Mult-VAE)."),
    min_user_items: int = typer.Option(1, help="Минимум items на пользователя."),
) -> None:
    """Только подготовить sparse user-history без обучения."""
    build_user_history_dataset(
        day=day,
        output_path=output_path,
        feature_dir=feature_dir,
        binary=binary,
        min_user_items=min_user_items,
    )


@app.command()
def train(
    day: int = typer.Option(
        ...,
        help=(
            "Последний день тренинга. Используется feature-снимок ``user_id-item_id/{day}.pq``, "
            "который агрегирует 30-дневное окно ``[day-29, day]`` (включительно). "
            "Чтобы избежать target-leak, тестовая оценка модели должна выполняться "
            "ТОЛЬКО на событиях за дни > day."
        ),
    ),
    artifacts_dir: Path = typer.Option(VAE_DIR, help="Папка для артефактов модели."),
    history_path: Path | None = typer.Option(
        None,
        help="Готовый sparse history parquet. Если None — будет собран на лету во временный файл.",
    ),
    feature_dir: Path = typer.Option(
        USER_ITEM_FEATURE_DIR, help="Папка с фичами по парам (user_id, item_id)."
    ),
    binary: bool = typer.Option(True, help="Бинаризовать веса (как в оригинальном Mult-VAE)."),
    min_user_items: int = typer.Option(5, help="Минимум items на пользователя."),
    encoder_dims: str = typer.Option(
        "600,200", help="Скрытые размерности энкодера, через запятую (последняя — латент)."
    ),
    dropout: float = typer.Option(0.5, help="Дропаут на входной user-history."),
    beta: float = typer.Option(0.2, help="Максимальный вес KL-члена."),
    epochs: int = typer.Option(30, help="Количество эпох."),
    batch_size: int = typer.Option(256, help="Размер батча."),
    lr: float = typer.Option(1e-3, help="Learning rate."),
    weight_decay: float = typer.Option(0.0, help="L2-регуляризация Adam."),
    total_anneal_steps: int = typer.Option(200_000, help="Шаги KL-annealing warm-up."),
    kl_schedule: str = typer.Option("linear", help="Форма KL-annealing: 'linear' или 'cosine'."),
    use_lr_scheduler: bool = typer.Option(
        True,
        help="Использовать torch CosineAnnealingLR для learning rate (lr -> lr/100).",
    ),
    num_workers: int = typer.Option(0, help="DataLoader workers (Windows: оставьте 0)."),
    device: str | None = typer.Option(None, help="cpu / cuda / None (auto)."),
    seed: int = typer.Option(42, help="Seed."),
) -> None:
    """Полный пайплайн: подготовка history (если нужно) + обучение Mult-VAE."""
    enc_dims = [int(x) for x in encoder_dims.split(",") if x.strip()]

    cleanup_history = False
    if history_path is None:
        # Временный файл живет только пока идет обучение.
        tmp_dir = Path(tempfile.mkdtemp(prefix="vae_history_"))
        history_path = tmp_dir / "history.pq"
        cleanup_history = True

    try:
        history_path, item_to_idx, user_to_idx = build_user_history_dataset(
            day=day,
            output_path=history_path,
            feature_dir=feature_dir,
            binary=binary,
            min_user_items=min_user_items,
        )

        train_vae(
            history_path=history_path,
            item_to_idx=item_to_idx,
            user_to_idx=user_to_idx,
            artifacts_dir=artifacts_dir,
            encoder_dims=enc_dims,
            dropout=dropout,
            beta=beta,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            total_anneal_steps=total_anneal_steps,
            kl_schedule=kl_schedule,
            use_lr_scheduler=use_lr_scheduler,
            num_workers=num_workers,
            device=device,
            seed=seed,
        )
    finally:
        if cleanup_history and history_path.parent.exists():
            shutil.rmtree(history_path.parent, ignore_errors=True)


if __name__ == "__main__":
    app()
