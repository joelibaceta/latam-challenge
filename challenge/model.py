import pandas as pd
import pickle
import os
import numpy as np
from datetime import datetime
from typing import Tuple, Union, List


class DelayModel:
    """
    A model for predicting flight delays based on historical data.

    Attributes:
        _model (XGBClassifier): The trained XGBoost model.
        _high_season_ranges (List[Tuple[datetime, datetime]]): Date ranges for determining high season.
        _threshold_in_minutes (int): Threshold to classify delays (in minutes).
        _features_cols (List[str]): Selected feature columns for training and prediction.
        _model_path (str): Path to save or load the model.
    """

    def __init__(self) -> None:
        """
        Initializes the DelayModel with default configurations.
        """
        self._model = None  # Model should be saved in this attribute.
        self._high_season_ranges: List[Tuple[datetime, datetime]] = [
            (datetime(1, 12, 15), datetime(1, 12, 31)),
            (datetime(1, 1, 1), datetime(1, 3, 3)),
            (datetime(1, 7, 15), datetime(1, 7, 31)),
            (datetime(1, 9, 11), datetime(1, 9, 30)),
        ]
        self._threshold_in_minutes: int = 15
        self._features_cols: List[str] = [
            "OPERA_Latin American Wings",
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air",
        ]
        self._model_path: str = "xgboost_model.pkl"

    def _is_high_season(self, date: str) -> int:
        """
        Determines if a given date falls within high season.

        Args:
            date (str): Date in the format '%Y-%m-%d %H:%M:%S'.

        Returns:
            int: 1 if the date is in high season, 0 otherwise.
        """
        year = int(date.split("-")[0])
        date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
        ranges = [
            (start.replace(year=year), end.replace(year=year))
            for start, end in self._high_season_ranges
        ]
        return int(any(start <= date <= end for start, end in ranges))

    def _get_min_diff(self, row: pd.Series) -> float:
        """
        Calculates the difference in minutes between two timestamps.

        Args:
            row (pd.Series): A row of data containing 'Fecha-O' and 'Fecha-I'.

        Returns:
            float: Difference in minutes between 'Fecha-O' and 'Fecha-I'.
        """
        fecha_o = datetime.strptime(row["Fecha-O"], "%Y-%m-%d %H:%M:%S")
        fecha_i = datetime.strptime(row["Fecha-I"], "%Y-%m-%d %H:%M:%S")
        return ((fecha_o - fecha_i).total_seconds()) / 60

    def _get_period_day(self, date: str) -> str:
        """
        Classifies the time of day into morning, afternoon, or night.

        Args:
            date (str): Date in the format '%Y-%m-%d %H:%M:%S'.

        Returns:
            str: 'morning', 'afternoon', or 'night'.
        """
        time = datetime.strptime(date, "%Y-%m-%d %H:%M:%S").time()
        if (
            datetime.strptime("05:00", "%H:%M").time()
            <= time
            <= datetime.strptime("11:59", "%H:%M").time()
        ):
            return "morning"
        elif (
            datetime.strptime("12:00", "%H:%M").time()
            <= time
            <= datetime.strptime("18:59", "%H:%M").time()
        ):
            return "afternoon"
        return "night"

    def preprocess(
        self, data: pd.DataFrame, target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepares raw data for training or prediction.

        Args:
            data (pd.DataFrame): Raw data.
            target_column (str, optional): Target column name. If provided, the target is returned.

        Returns:
            Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
                - Features and target if target_column is specified.
                - Features only otherwise.
        """
        data["high_season"] = data["Fecha-I"].apply(self._is_high_season)
        data["min_diff"] = data.apply(self._get_min_diff, axis=1)
        data["period_day"] = data["Fecha-I"].apply(self._get_period_day)
        data["delay"] = np.where(data["min_diff"] > self._threshold_in_minutes, 1, 0)

        # Shuffle and select necessary columns
        data = data[["OPERA", "MES", "TIPOVUELO", "SIGLADES", "DIANOM", "delay"]]

        features = pd.concat(
            [
                pd.get_dummies(data["OPERA"], prefix="OPERA"),
                pd.get_dummies(data["TIPOVUELO"], prefix="TIPOVUELO"),
                pd.get_dummies(data["MES"], prefix="MES"),
            ],
            axis=1,
        )

        # Ensure all required features are present
        for col in self._features_cols:
            if col not in features.columns:
                features[col] = 0
        features = features[self._features_cols]

        if target_column:
            target = data[[target_column]]
            return features, target
        return features

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Trains the XGBoost model with the provided features and target.

        Args:
            features (pd.DataFrame): Preprocessed feature data.
            target (pd.DataFrame): Target values.
        """
        from xgboost import XGBClassifier
        from sklearn.model_selection import train_test_split

        x_train, x_test, y_train, y_test = train_test_split(
            features, target, test_size=0.33, random_state=42
        )
        if isinstance(y_train, pd.DataFrame):
            y_train = y_train.squeeze()
        n_y0 = len(y_train[y_train == 0])
        n_y1 = len(y_train[y_train == 1])
        scale = n_y0 / n_y1

        self._model = XGBClassifier(
            random_state=1, learning_rate=0.01, scale_pos_weight=scale
        )
        self._model.fit(x_train, y_train)

        # Save the model
        with open(self._model_path, "wb") as file:
            pickle.dump(self._model, file)

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predicts delays using the trained model.

        Args:
            features (pd.DataFrame): Preprocessed feature data.

        Returns:
            List[int]: Predicted delay labels.
        """
        if not self._model:
            if not os.path.exists(self._model_path):
                raise FileNotFoundError(
                    f"Model not found at {self._model_path}. Train and save the model first."
                )
            with open(self._model_path, "rb") as file:
                self._model = pickle.load(file)
        features = features[self._features_cols]
        return self._model.predict(features).tolist()
