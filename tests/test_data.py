import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from src.dataset import _ndcg_at_k_loop, ndcg_at_k


NDCG_KWARGS = dict(
    relevancy_col="relevancy",
    true_items_col="true_items",
    predicted_items_col="predicted_items",
    predicted_score_col="predicted_scores",
)


def _make_simple_df() -> pl.DataFrame:
    """Простой датасет из 2 пользователей (из SVD-ноутбука)."""
    return pl.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "true_items": [["A", "B", "C"], ["A", "B", "C"]],
            "relevancy": [[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]],
            "predicted_items": [["A", "B", "C"], ["C", "B", "A"]],
            "predicted_scores": [[0.9, 0.8, 0.7], [0.9, 0.8, 0.7]],
        }
    )


def _make_edge_cases_df() -> pl.DataFrame:
    """Датасет с граничными случаями."""
    return pl.DataFrame(
        {
            "user_id": ["u1", "u2", "u3", "u4"],
            "true_items": [
                ["A", "B", "C"],
                ["X", "Y"],
                ["A", "A", "B"],
                ["A", "B", "C", "D", "E"],
            ],
            "relevancy": [
                [3.0, 2.0, 1.0],
                [5.0, 3.0],
                [3.0, 1.0, 2.0],
                [5.0, 4.0, 3.0, 2.0, 1.0],
            ],
            "predicted_items": [
                ["A", "B", "C"],
                ["A", "B", "C"],
                ["A", "B", "D"],
                ["E", "D", "C", "B", "A"],
            ],
            "predicted_scores": [
                [0.9, 0.8, 0.7],
                [0.9, 0.8, 0.7],
                [0.9, 0.8, 0.7],
                [0.9, 0.8, 0.7, 0.6, 0.5],
            ],
        }
    )


def _assert_ndcg_equal(df: pl.DataFrame, top_k: int = 3) -> None:
    """Проверяет, что vectorized и loop-версии дают одинаковый результат."""
    result_new = ndcg_at_k(df, **NDCG_KWARGS, top_k=top_k).sort("user_id")
    result_old = _ndcg_at_k_loop(df, **NDCG_KWARGS, top_k=top_k).sort("user_id")
    assert_frame_equal(result_new, result_old, abs_tol=1e-10)


class TestNdcgAtK:
    def test_perfect_ranking(self):
        df = _make_simple_df()
        result = ndcg_at_k(df, **NDCG_KWARGS, top_k=3).sort("user_id")
        u1_ndcg = result.filter(pl.col("user_id") == "u1")["ndcg"].item()
        assert u1_ndcg == pytest.approx(1.0)

    def test_imperfect_ranking(self):
        df = _make_simple_df()
        result = ndcg_at_k(df, **NDCG_KWARGS, top_k=3).sort("user_id")
        u2_ndcg = result.filter(pl.col("user_id") == "u2")["ndcg"].item()
        assert 0.0 < u2_ndcg < 1.0

    def test_matches_loop_simple(self):
        _assert_ndcg_equal(_make_simple_df(), top_k=3)

    def test_matches_loop_edge_cases(self):
        _assert_ndcg_equal(_make_edge_cases_df(), top_k=3)

    def test_matches_loop_top_k_less_than_predictions(self):
        _assert_ndcg_equal(_make_edge_cases_df(), top_k=2)

    def test_matches_loop_top_k_greater_than_predictions(self):
        _assert_ndcg_equal(_make_simple_df(), top_k=10)

    def test_no_relevant_items_gives_zero(self):
        """Если ни один предсказанный айтем не релевантен, NDCG = 0."""
        df = _make_edge_cases_df()
        result = ndcg_at_k(df, **NDCG_KWARGS, top_k=3).sort("user_id")
        u2_ndcg = result.filter(pl.col("user_id") == "u2")["ndcg"].item()
        assert u2_ndcg == 0.0

    def test_duplicate_truth_items(self):
        """Дубликаты в truth — берётся max relevancy."""
        df = _make_edge_cases_df()
        result_new = ndcg_at_k(df, **NDCG_KWARGS, top_k=3).sort("user_id")
        result_old = _ndcg_at_k_loop(df, **NDCG_KWARGS, top_k=3).sort("user_id")
        u3_new = result_new.filter(pl.col("user_id") == "u3")["ndcg"].item()
        u3_old = result_old.filter(pl.col("user_id") == "u3")["ndcg"].item()
        assert u3_new == pytest.approx(u3_old, abs=1e-10)

    def test_matches_loop_random(self):
        """Случайный датасет из 200 пользователей."""
        rng = np.random.RandomState(42)
        n_users = 200
        users = [f"user_{i}" for i in range(n_users)]
        all_items = [f"item_{i}" for i in range(50)]

        true_items_list = []
        relevancy_list = []
        pred_items_list = []
        pred_scores_list = []

        for _ in range(n_users):
            n_true = rng.randint(3, 15)
            n_pred = rng.randint(5, 20)
            t_items = rng.choice(all_items, size=n_true, replace=True).tolist()
            rels = rng.exponential(2.0, size=n_true).tolist()
            p_items = rng.choice(all_items, size=n_pred, replace=False).tolist()
            scores = sorted(rng.uniform(0, 1, size=n_pred).tolist(), reverse=True)

            true_items_list.append(t_items)
            relevancy_list.append(rels)
            pred_items_list.append(p_items)
            pred_scores_list.append(scores)

        df = pl.DataFrame(
            {
                "user_id": users,
                "true_items": true_items_list,
                "relevancy": relevancy_list,
                "predicted_items": pred_items_list,
                "predicted_scores": pred_scores_list,
            }
        )
        _assert_ndcg_equal(df, top_k=10)
