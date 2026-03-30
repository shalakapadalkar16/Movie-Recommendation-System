import pandas as pd


class PopularityRecommender:
    """
    Simple non-personalized popularity-based recommender.

    This recommender ranks movies by the total number of ratings they have received.
    It is useful as:
    - a cold-start baseline
    - a benchmark for more advanced recommenders
    - a "popular now" or "top movies" style module
    """

    def __init__(self) -> None:
        self.popularity_df: pd.DataFrame | None = None

    def fit(self, ratings: pd.DataFrame, movies: pd.DataFrame) -> None:
        """
        Build the popularity table from ratings and movie metadata.

        Parameters
        ----------
        ratings : pd.DataFrame
            Must contain at least: userId, movieId, rating, timestamp
        movies : pd.DataFrame
            Must contain at least: movieId, title, genres
        """
        movie_rating_counts = (
            ratings.groupby("movieId")
            .size()
            .reset_index(name="rating_count")
        )

        self.popularity_df = (
            movie_rating_counts
            .merge(movies, on="movieId", how="left")
            .sort_values(by="rating_count", ascending=False)
            .reset_index(drop=True)
        )

    def recommend(self, top_k: int = 10) -> pd.DataFrame:
        """
        Return the top-k most popular movies.

        Parameters
        ----------
        top_k : int
            Number of recommendations to return.

        Returns
        -------
        pd.DataFrame
            Top-k movies ranked by rating_count.
        """
        if self.popularity_df is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        return self.popularity_df.head(top_k).copy()
    
class WeightedPopularityRecommender:
    """
    Popularity recommender using IMDb-style weighted rating.
    """

    def __init__(self, min_votes: int = 1000) -> None:
        self.min_votes = min_votes
        self.popularity_df: pd.DataFrame | None = None

    def fit(self, ratings: pd.DataFrame, movies: pd.DataFrame) -> None:
        movie_stats = (
            ratings.groupby("movieId")
            .agg(
                rating_count=("rating", "count"),
                avg_rating=("rating", "mean")
            )
            .reset_index()
        )

        C = movie_stats["avg_rating"].mean()
        m = self.min_votes

        # IMDb weighted score
        movie_stats["weighted_score"] = (
            (movie_stats["rating_count"] / (movie_stats["rating_count"] + m)) * movie_stats["avg_rating"]
            + (m / (movie_stats["rating_count"] + m)) * C
        )

        self.popularity_df = (
            movie_stats
            .merge(movies, on="movieId", how="left")
            .sort_values(by="weighted_score", ascending=False)
            .reset_index(drop=True)
        )

    def recommend(self, top_k: int = 10) -> pd.DataFrame:
        if self.popularity_df is None:
            raise ValueError("Model has not been fitted yet.")

        return self.popularity_df.head(top_k).copy()
    

import pandas as pd


class TrendingRecommender:
    """
    Window-based trending recommender.

    This recommender ranks movies by the number of ratings received
    within a recent time window (for example, last 30, 90, or 365 days).

    It is useful for answering:
    - What is trending right now?
    - What has become popular recently?
    """

    def __init__(self, window_days: int = 365) -> None:
        self.window_days = window_days
        self.trending_df: pd.DataFrame | None = None
        self.reference_time: pd.Timestamp | None = None

    def fit(self, ratings: pd.DataFrame, movies: pd.DataFrame) -> None:
        """
        Build the trending table from recent ratings only.

        Parameters
        ----------
        ratings : pd.DataFrame
            Must contain: userId, movieId, rating, timestamp
        movies : pd.DataFrame
            Must contain: movieId, title, genres
        """
        ratings_copy = ratings.copy()

        # Make sure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(ratings_copy["timestamp"]):
            ratings_copy["timestamp"] = pd.to_datetime(ratings_copy["timestamp"], unit="s")

        # Use the latest timestamp in the dataset as the reference point
        self.reference_time = ratings_copy["timestamp"].max()

        # Define start of recent window
        window_start = self.reference_time - pd.Timedelta(days=self.window_days)

        # Keep only ratings inside the window
        recent_ratings = ratings_copy[ratings_copy["timestamp"] >= window_start]

        # Count how many recent ratings each movie received
        recent_counts = (
            recent_ratings.groupby("movieId")
            .size()
            .reset_index(name="recent_rating_count")
        )

        self.trending_df = (
            recent_counts
            .merge(movies, on="movieId", how="left")
            .sort_values(by="recent_rating_count", ascending=False)
            .reset_index(drop=True)
        )

    def recommend(self, top_k: int = 10) -> pd.DataFrame:
        """
        Return the top-k trending movies in the recent window.
        """
        if self.trending_df is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        return self.trending_df.head(top_k).copy()
    

class TrendingWeightedRecommender:
    """
    Window-based trending recommender with rating quality.

    Combines:
    - recent interaction count
    - recent average rating
    using a weighted (IMDb-style) score
    """

    def __init__(self, window_days: int = 90, min_votes: int = 50) -> None:
        self.window_days = window_days
        self.min_votes = min_votes
        self.trending_df: pd.DataFrame | None = None

    def fit(self, ratings: pd.DataFrame, movies: pd.DataFrame) -> None:
        ratings_copy = ratings.copy()

        # ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(ratings_copy["timestamp"]):
            ratings_copy["timestamp"] = pd.to_datetime(ratings_copy["timestamp"], unit="s")

        reference_time = ratings_copy["timestamp"].max()
        window_start = reference_time - pd.Timedelta(days=self.window_days)

        recent_ratings = ratings_copy[ratings_copy["timestamp"] >= window_start]

        # compute recent stats
        movie_stats = (
            recent_ratings.groupby("movieId")
            .agg(
                recent_rating_count=("rating", "count"),
                recent_avg_rating=("rating", "mean")
            )
            .reset_index()
        )

        # global average (within window)
        C = movie_stats["recent_avg_rating"].mean()
        m = self.min_votes

        # weighted score
        movie_stats["trending_score"] = (
            (movie_stats["recent_rating_count"] / (movie_stats["recent_rating_count"] + m)) * movie_stats["recent_avg_rating"]
            + (m / (movie_stats["recent_rating_count"] + m)) * C
        )

        self.trending_df = (
            movie_stats
            .merge(movies, on="movieId", how="left")
            .sort_values(by="trending_score", ascending=False)
            .reset_index(drop=True)
        )

    def recommend(self, top_k: int = 10) -> pd.DataFrame:
        if self.trending_df is None:
            raise ValueError("Model not fitted yet")

        return self.trending_df.head(top_k).copy()
    

import numpy as np
import pandas as pd


class TimeDecayWeightedTrendingRecommender:
    """
    Time-decayed trending recommender with rating quality.

    Each rating is weighted by recency using exponential decay:
        weight = exp(-lambda_decay * age_in_days)

    Then for each movie, we compute:
    - decayed_count
    - decayed_avg_rating

    Finally, we apply a Bayesian-style weighted score to combine:
    - recent weighted volume
    - recent weighted quality
    - confidence smoothing
    """

    def __init__(self, lambda_decay: float = 0.01, min_weight: float = 50.0) -> None:
        self.lambda_decay = lambda_decay
        self.min_weight = min_weight
        self.trending_df: pd.DataFrame | None = None
        self.reference_time: pd.Timestamp | None = None

    def fit(self, ratings: pd.DataFrame, movies: pd.DataFrame) -> None:
        ratings_copy = ratings.copy()

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(ratings_copy["timestamp"]):
            ratings_copy["timestamp"] = pd.to_datetime(ratings_copy["timestamp"], unit="s")

        # Use the latest timestamp in dataset as "now"
        self.reference_time = ratings_copy["timestamp"].max()

        # Age of each interaction in days
        ratings_copy["age_in_days"] = (
            self.reference_time - ratings_copy["timestamp"]
        ).dt.total_seconds() / (24 * 60 * 60)

        # Exponential time decay
        ratings_copy["decay_weight"] = np.exp(-self.lambda_decay * ratings_copy["age_in_days"])

        # Weighted rating contribution
        ratings_copy["weighted_rating"] = ratings_copy["rating"] * ratings_copy["decay_weight"]

        # Aggregate per movie
        movie_stats = (
            ratings_copy.groupby("movieId")
            .agg(
                decayed_count=("decay_weight", "sum"),
                decayed_rating_sum=("weighted_rating", "sum")
            )
            .reset_index()
        )

        # Decayed average rating
        movie_stats["decayed_avg_rating"] = (
            movie_stats["decayed_rating_sum"] / movie_stats["decayed_count"]
        )

        # Global decayed average rating
        C = ratings_copy["weighted_rating"].sum() / ratings_copy["decay_weight"].sum()
        m = self.min_weight

        # Bayesian-style weighted score
        movie_stats["trending_score"] = (
            (movie_stats["decayed_count"] / (movie_stats["decayed_count"] + m)) * movie_stats["decayed_avg_rating"]
            + (m / (movie_stats["decayed_count"] + m)) * C
        )

        self.trending_df = (
            movie_stats
            .merge(movies, on="movieId", how="left")
            .sort_values(by="trending_score", ascending=False)
            .reset_index(drop=True)
        )

    def recommend(self, top_k: int = 10) -> pd.DataFrame:
        if self.trending_df is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        return self.trending_df.head(top_k).copy()
    