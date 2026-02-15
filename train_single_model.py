import pandas as pd
import numpy as np

try:
    import joblib
except ImportError:
    from sklearn.externals import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from clearml import Task, Logger, OutputModel
import argparse


def get_model_params(model: str) -> dict[str, dict[str, int | float]]:
    cfg = {
        "GradientBoostingRegressor": {
            "n_estimators": 200,
            "learning_rate": 0.1,
            "max_depth": 5,
            "random_state": 42,
        },
        "KNeighborsRegressor": {"n_neighbors": 5},
        "LinearRegression": {},
    }
    return cfg[model]


def get_model(
    model: str,
) -> GradientBoostingRegressor | KNeighborsRegressor | LinearRegression:
    models = {
        "GradientBoostingRegressor": GradientBoostingRegressor,
        "KNeighborsRegressor": KNeighborsRegressor,
        "LinearRegression": LinearRegression,
    }

    return models[model]


# Оценка модели и логирование метрик
def print_metrics(
    name: str, y_true: pd.Series, y_pred: np.ndarray, logger: Logger, model: OutputModel
):
    logger.report_scalar(
        name, "RMSE", value=mean_squared_error(y_true, y_pred), iteration=0
    )
    logger.report_scalar(
        name, "MAE", value=mean_absolute_error(y_true, y_pred), iteration=0
    )
    logger.report_scalar(name, "R2", value=r2_score(y_true, y_pred), iteration=0)

    model.report_scalar(
        name, "RMSE", value=mean_squared_error(y_true, y_pred), iteration=0
    )
    model.report_scalar(
        name, "MAE", value=mean_absolute_error(y_true, y_pred), iteration=0
    )
    model.report_scalar(name, "R2", value=r2_score(y_true, y_pred), iteration=0)


def main(model: str):
    # Подключение ClearML для логирования
    task: Task = Task.init(
        project_name="uber", reuse_last_task_id=False, output_uri=True
    )
    task.execute_remotely(queue_name='ya_queue')
    logger: Logger = task.get_logger()
    logger.report_text(f"Train {model} model")

    # Загрузка и предобработка данных
    df = pd.read_csv("uber.csv")

    # Логирование артефакта
    task.register_artifact(
        name="uber_fares_dataset",
        artifact=df,
        metadata={
            "description": "Uber fares dataset with pickup and dropoff locations, fare amount, and passenger count."
        },
    )

    # Удаление пропусков и выбросов
    df = df.dropna()
    df = df[
        (df["fare_amount"] > 0)
        & (df["passenger_count"] > 0)
        & (df["passenger_count"] <= 6)
    ]

    # Выбор признаков
    X = df[
        [
            "pickup_latitude",
            "pickup_longitude",
            "dropoff_latitude",
            "dropoff_longitude",
            "passenger_count",
        ]
    ]
    y = df["fare_amount"]

    stats = X.describe()

    # Логирование статистик датасета после обработки как артефакта
    task.register_artifact(
        name="dataset_statistics",
        artifact=stats,
        metadata={"description": "Dataset statistics after post-processing."},
    )

    # Логирование статистик датасета после обработки как таблицы
    logger.report_table(
        title="Datasets statts", series="Stats", iteration=0, table_plot=stats
    )
    params = {
        "data": {
            "test_size": 0.2,
            "random_state": 42,
        },
        "model": {"name": model, "params": get_model_params(model)},
    }

    logger.report_text("Params")
    logger.report_text(params)

    # Логирование параметров
    task.connect(params)

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, **params["data"])

    # Создание объекта OutputModel
    output_model = OutputModel(
        task=task,
        framework="ScikitLearn",
        name=model,
        comment=f"{model} for uber",
        tags=["uber", "reg_model"],
    )
    reg = get_model(model)(**get_model_params(model))
    reg.fit(X_train, y_train)

    # Сохранение модели и тестирование
    joblib.dump(reg, f"{model}.pkl", compress=True)
    predicts = reg.predict(X_test)
    output_model.update_weights(f"{model}.pkl")

    # Вывод метрик
    print_metrics(model, y_test, predicts, logger, output_model)
    print_metrics(model, y_test, predicts, logger, output_model)
    print_metrics(model, y_test, predicts, logger, output_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        choices=[
            "GradientBoostingRegressor",
            "KNeighborsRegressor",
            "LinearRegression",
        ],
        required=True,
    )
    args = parser.parse_args()
    main(args.model)
