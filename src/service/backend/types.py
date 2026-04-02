from typing import Literal

from src.modeling.models import ModelsType

ActionType = Literal["view", "click", "clickout", "like"]
Subdomain = Literal["u2i", "i2i", "catalog", "search", "other"]
Os = Literal["android", "ios", "other"]
ModelKey = ModelsType
