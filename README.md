# Проект: Тема 31. «Создание рекомендательной системы товаров для ретейла».

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Команда

*   **Куратор:** Руслан Каюмов
*   **Участники:**
    *   Александр Мишин
    *   Александр Викторов
    *   Светлана Козина
    *   Мария Кузнецова

---

## Техническая настройка и окружение


*   **Версия Python:** 3.13.2
*   **Менеджер пакетов:** [uv](https://docs.astral.sh/uv/). Современный и быстрый инструмент для управления зависимостями проекта.
*   **Линтер и форматирование кода:** [Ruff](https://docs.astral.sh/ruff/). Отвечает за проверку стиля кода. Рекомендуется установить [расширение для VSCode](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff).

---

## План работы

### Этап 1: Установочный

*    Изучение предметной области.
*    Просмотр [лекций по рекомендательным системам.](http://wiki.cs.hse.ru/%D0%98%D0%98_%D0%A0%D0%B5%D0%BA%D0%BE%D0%BC%D0%B5%D0%BD%D0%B4%D0%B0%D1%82%D0%B5%D0%BB%D1%8C%D0%BD%D1%8B%D0%B5_%D1%81%D0%B8%D1%81%D1%82%D0%B5%D0%BC%D1%8B_(2024-25,_3-4_%D0%BC%D0%BE%D0%B4%D1%83%D0%BB%D0%B8))
*   Организация рабочего репозитория команды.



### Этап 2: Разведочный анализ данных (EDA) (до 20-х чисел октября 2025)

*   Поиск датасетов для e-commerce.
*   Выбор и утверждение основного датасета для работы.
*   Определение ключевых метрик.
*   Проведение разведочного анализа (EDA).
*   Визуализация основных свойств данных.
*   Предобработка и очистка данных.
*   Определение концепции сервиса.

### Этап 3: Базовые и линейные ML-модели (до конца ноября 2025)

*   Реализация неперсонализированных бейзлайнов (топ-популярных).
*   Реализация классических моделей коллаборативной фильтрации.
*   Построение модели на основе матричной факторизации.
*   Оценка качества моделей по выбранным метрикам.

### Этап 4: Нелинейные ML-модели (до середины января 2026)

*   Разработка архитектуры и начало реализации прототипа сервиса на основе лучших базовых моделей.
*   Реализация нелинейных моделей ML.
*   Проведение экспериментов с переранжированием рекомендаций.
*   Проведение сравнительного анализа всех реализованных моделей.

### Промежуточная защита (конец января 2026)

*   Подготовка отчета и презентации.

### Этап 5: Создание сервиса с имплементацией лучшего ML-решения (до середины февраля 2026)

### Этап 6: Внедрение Deep Learning (до конца марта - начала апреля 2026)


### Этап 7: Доработка задачи: улучшение сервисной части по обратной связи от команды курса П.Р.; тюнинг DL-моделей и выбор лучшего решения overall (до середины мая 2026)

### Возможные направления для исследования:
  *  **Графовые нейросети:** Применение графовых моделей.
  *  **Интерпретируемость:** Реализация подходов для объяснения сгенерированных рекомендаций.

### Финальный этап: Подготовка к защите (июнь 2026)

  *   Завершение оформления репозитория, кода и документации.
  *   Подготовка финального отчета и презентации для защиты проекта.

---

## Структура проекта

Проект организован с использованием шаблона `cookiecutter-data-science`.

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         dreamteam_recsys and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── src   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes dreamteam_recsys a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

