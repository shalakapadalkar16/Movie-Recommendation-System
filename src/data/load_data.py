import pandas as pd
from pathlib import Path


def load_ratings(data_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(data_path) / "ratings.csv")


def load_movies(data_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(data_path) / "movies.csv")


def load_tags(data_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(data_path) / "tags.csv")


def load_links(data_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(data_path) / "links.csv")