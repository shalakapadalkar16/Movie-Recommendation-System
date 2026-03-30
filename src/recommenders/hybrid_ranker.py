import pandas as pd
from typing import Optional

from src.recommenders.popularity_recommender import (
    PopularityRecommender,
    WeightedPopularityRecommender,
    TrendingRecommender,
    TimeDecayWeightedTrendingRecommender,
)


class HybridRankingRecommender:
    """
    Hybrid ranker that combines multiple popularity/trending signals.

    Signals used:
    - all-time popularity count
    - all-time weighted popularity
    - recent window-based trending
    - time-decayed trending with rating quality
    """

    def __init__(
        self,
        trending_window_days: int = 90,
        lambda_decay: float = 0.01,
        min_votes: int = 1000,
        min_weight: float = 50.0,
        w_popularity: float = 0.20,
        w_weighted_popularity: float = 0.30,
        w_recent_trending: float = 0.20,
        w_time_decay: float = 0.30,
    ) -> None:
        self.trending_window_days = trending_window_days
        self.lambda_decay = lambda_decay
        self.min_votes = min_votes
        self.min_weight = min_weight

        self.w_popularity = w_popularity
        self.w_weighted_popularity = w_weighted_popularity
        self.w_recent_trending = w_recent_trending
        self.w_time_decay = w_time_decay

        self.ranking_df: Optional[pd.DataFrame] = None

    @staticmethod
    def _min_max_normalize(series: pd.Series) -> pd.Series:
        min_val = series.min()
        max_val = series.max()

        if max_val == min_val:
            return pd.Series(0.0, index=series.index)

        return (series - min_val) / (max_val - min_val)

    def fit(self, ratings: pd.DataFrame, movies: pd.DataFrame) -> None:
        # 1. Train component recommenders
        pop_rec = PopularityRecommender()
        pop_rec.fit(ratings, movies)

        weighted_pop_rec = WeightedPopularityRecommender(min_votes=self.min_votes)
        weighted_pop_rec.fit(ratings, movies)

        recent_trend_rec = TrendingRecommender(window_days=self.trending_window_days)
        recent_trend_rec.fit(ratings, movies)

        time_decay_rec = TimeDecayWeightedTrendingRecommender(
            lambda_decay=self.lambda_decay,
            min_weight=self.min_weight,
        )
        time_decay_rec.fit(ratings, movies)

        # 2. Extract relevant columns
        popularity_df = pop_rec.popularity_df[["movieId", "rating_count"]].copy()

        weighted_popularity_df = weighted_pop_rec.popularity_df[
            ["movieId", "weighted_score"]
        ].copy()

        recent_trending_df = recent_trend_rec.trending_df[
            ["movieId", "recent_rating_count"]
        ].copy()

        time_decay_df = time_decay_rec.trending_df[
            ["movieId", "trending_score"]
        ].copy().rename(columns={"trending_score": "time_decay_score"})

        # 3. Merge all signals
        hybrid_df = popularity_df.merge(
            weighted_popularity_df, on="movieId", how="outer"
        ).merge(
            recent_trending_df, on="movieId", how="outer"
        ).merge(
            time_decay_df, on="movieId", how="outer"
        )

        hybrid_df = hybrid_df.merge(movies, on="movieId", how="left")

        # 4. Fill missing values with 0 for signals
        signal_cols = [
            "rating_count",
            "weighted_score",
            "recent_rating_count",
            "time_decay_score",
        ]
        hybrid_df[signal_cols] = hybrid_df[signal_cols].fillna(0)

        # 5. Normalize each signal
        hybrid_df["popularity_norm"] = self._min_max_normalize(hybrid_df["rating_count"])
        hybrid_df["weighted_popularity_norm"] = self._min_max_normalize(hybrid_df["weighted_score"])
        hybrid_df["recent_trending_norm"] = self._min_max_normalize(hybrid_df["recent_rating_count"])
        hybrid_df["time_decay_norm"] = self._min_max_normalize(hybrid_df["time_decay_score"])

        # 6. Final hybrid score
        hybrid_df["hybrid_score"] = (
            self.w_popularity * hybrid_df["popularity_norm"]
            + self.w_weighted_popularity * hybrid_df["weighted_popularity_norm"]
            + self.w_recent_trending * hybrid_df["recent_trending_norm"]
            + self.w_time_decay * hybrid_df["time_decay_norm"]
        )

        # 7. Sort
        self.ranking_df = hybrid_df.sort_values(
            by="hybrid_score", ascending=False
        ).reset_index(drop=True)

    def recommend(self, top_k: int = 10) -> pd.DataFrame:
        if self.ranking_df is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        cols_to_show = [
            "movieId",
            "title",
            "genres",
            "rating_count",
            "weighted_score",
            "recent_rating_count",
            "time_decay_score",
            "hybrid_score",
        ]

        available_cols = [col for col in cols_to_show if col in self.ranking_df.columns]
        return self.ranking_df[available_cols].head(top_k).copy()