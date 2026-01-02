from typing import Literal, cast

from loguru import logger
import typer

from src.modeling.models import ModelsType, create_model
from src.modeling.sharing import InferenceData

app = typer.Typer()


@app.command()
def main(
    user_id: int,
    item_id: int,
    model_params: dict | None = None,
    model_key: ModelsType = typer.Option(..., help="Key of the model to use (e.g. 'dummy')"),
):
    """
    CLI tool to run model prediction manually.
    Useful for debugging and verifying models without starting the full service.
    """
    # CLI утилита для запуска предсказания модели вручную.
    # Удобно для отладки и проверки моделей без запуска всего сервиса.
    input_data = InferenceData(
        user_id=user_id,
        item_id=item_id,
        model_params=model_params,
    )

    logger.info(f"Initializing model '{model_key}'...")
    try:
        # Use Factory to create configured model instance
        model = create_model(model_key)
        # Manually trigger load for CLI usage
        model.loads()
    except ValueError as e:
        logger.error(str(e))
        raise typer.Exit(code=1)

    logger.info("Running prediction...")
    result = model.predict(input_data)

    logger.info(f"Prediction result: {result}")
    return result


if __name__ == "__main__":
    app()
