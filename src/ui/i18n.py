"""Small translation layer for the Streamlit dashboard."""

from __future__ import annotations

from collections.abc import Mapping


LANGUAGE_LABELS = {"en": "English", "ru": "Русский"}
LANGUAGE_CODES_BY_LABEL = {label: code for code, label in LANGUAGE_LABELS.items()}

PAGE_KEYS = (
    "overview",
    "single_client",
    "batch_scoring",
    "data_explorer",
    "model_explainability",
    "model_performance",
)

TEXT: dict[str, dict[str, str]] = {
    "en": {
        "nav.overview": "🏠 Overview",
        "nav.single_client": "👤 Single Client Scoring",
        "nav.batch_scoring": "📁 Batch Scoring",
        "nav.data_explorer": "📊 Data Explorer",
        "nav.model_explainability": "🧠 Model Explainability",
        "nav.model_performance": "⚙️ Model Performance",
        "sidebar.brand_caption": "Bank analyst dashboard",
        "sidebar.language": "Language",
        "sidebar.navigation": "Navigation",
        "sidebar.api_base_url": "API base URL",
        "sidebar.api_health": "API Health",
        "sidebar.api_unavailable": "API unavailable",
        "sidebar.model_loaded": "Model loaded",
        "sidebar.model_missing": "Model missing",
        "sidebar.model_missing_detail": "Model artifact is not available.",
        "sidebar.open_swagger": "Open Swagger",
        "page.overview.title": "Home Credit Default Risk",
        "page.overview.subtitle": "Decision-support tool for credit risk analysts.",
        "overview.total_applications": "Total Applications",
        "overview.default_rate": "Default Rate",
        "overview.model_auc": "Model AUC",
        "overview.threshold": "Current Threshold",
        "overview.project_goal": "Project Goal",
        "overview.project_goal_text": (
            "This dashboard estimates the probability that a client may have difficulty repaying a loan. "
            "The output is decision support for analysts, not an automatic credit decision."
        ),
        "overview.recommendation_review": "Recommendation: Review Required",
        "overview.data_scope": "Data Scope",
        "overview.data_scope_text": (
            "Main application features are combined with historical credit signals from the Home Credit dataset."
        ),
        "overview.target_note": "TARGET: 0 means repaid, 1 means repayment difficulty.",
        "page.single.title": "Single Client Scoring",
        "page.single.subtitle": "Estimate default probability for one applicant.",
        "single.standard_profile": "Standard profile",
        "single.stressed_profile": "Stressed profile",
        "single.client_profile": "Client Profile",
        "single.loan_parameters": "Loan Parameters",
        "single.external_history": "External Sources And Credit History",
        "single.calculate": "Calculate Risk",
        "field.age": "Age",
        "field.gender": "Gender",
        "field.contract_type": "Contract Type",
        "field.education": "Education",
        "field.family_status": "Family Status",
        "field.income_type": "Income Type",
        "field.occupation_type": "Occupation Type",
        "field.children_count": "Children Count",
        "field.family_members": "Family Members",
        "field.total_income": "Total Income",
        "field.credit_amount": "Credit Amount",
        "field.annuity": "Annuity",
        "field.goods_price": "Goods Price",
        "field.employment_years": "Employment Years",
        "result.probability": "Probability of Default",
        "result.waiting": "Waiting",
        "result.waiting_hint": "Run scoring to calculate risk.",
        "result.recommendation": "Recommendation",
        "result.increasing": "Risk Increasing Factors",
        "result.reducing": "Risk Reducing Factors",
        "result.coverage": "Model input coverage",
        "result.missing_features": (
            "{count} historical features were not entered manually and were handled by "
            "the model preprocessing pipeline."
        ),
        "result.extra_features": (
            "These profile fields are displayed for analyst context but are not used by the current top-N model: "
            "{features}."
        ),
        "risk.low": "Low Risk",
        "risk.medium": "Medium Risk",
        "risk.high": "High Risk",
        "risk.unknown": "Unknown Risk",
        "recommendation.low": "Acceptable risk profile",
        "recommendation.medium": "Manual review required",
        "recommendation.high": "High-risk application",
        "recommendation.unknown": "Review required",
        "page.batch.title": "Batch Scoring",
        "page.batch.subtitle": "Upload applications CSV and score multiple clients.",
        "batch.upload": "Upload applications CSV",
        "batch.callout": "Upload a CSV with Home Credit feature columns to run batch scoring.",
        "batch.preview": "Preview",
        "batch.run": "Run Batch Prediction",
        "batch.results": "Scoring Results",
        "batch.download": "Download predictions CSV",
        "batch.low": "Low",
        "batch.medium": "Medium",
        "batch.high": "High",
        "page.data.title": "Data Explorer",
        "page.data.subtitle": "Dataset quality, target balance and feature distributions.",
        "data.target_distribution": "Target distribution",
        "data.top_missing_values": "Top missing values",
        "data.credit_risk_correlations": "Credit-risk correlations",
        "data.pca_projection": "PCA projection",
        "data.missing_artifact": "Missing artifact: {path}",
        "data.missing_values_table": "Missing Values Table",
        "page.explain.title": "Model Explainability",
        "page.explain.subtitle": "Global feature importance and local explanation language.",
        "explain.feature_unavailable": "Feature importance artifact is not available.",
        "explain.global_importance": "Global Feature Importance",
        "page.performance.title": "Model Performance",
        "page.performance.subtitle": "Experiment metrics and threshold review panel.",
        "performance.results_unavailable": "Experiment results are not available.",
        "performance.experiment_table": "Experiment Table",
        "performance.threshold_analysis": "Threshold Analysis",
        "performance.decision_threshold": "Decision threshold",
        "performance.threshold_text": (
            "The committed experiment table stores aggregate metrics at the model evaluation threshold. "
            "The slider is included for analyst review workflow; exact threshold curves require saved "
            "validation probabilities."
        ),
        "performance.current_threshold": "Current threshold",
    },
    "ru": {
        "nav.overview": "🏠 Обзор",
        "nav.single_client": "👤 Скоринг клиента",
        "nav.batch_scoring": "📁 Пакетный скоринг",
        "nav.data_explorer": "📊 Анализ данных",
        "nav.model_explainability": "🧠 Интерпретация модели",
        "nav.model_performance": "⚙️ Качество модели",
        "sidebar.brand_caption": "Кабинет банковского аналитика",
        "sidebar.language": "Язык",
        "sidebar.navigation": "Навигация",
        "sidebar.api_base_url": "Адрес API",
        "sidebar.api_health": "Статус API",
        "sidebar.api_unavailable": "API недоступен",
        "sidebar.model_loaded": "Модель загружена",
        "sidebar.model_missing": "Модель не найдена",
        "sidebar.model_missing_detail": "Артефакт модели недоступен.",
        "sidebar.open_swagger": "Открыть Swagger",
        "page.overview.title": "Home Credit Default Risk",
        "page.overview.subtitle": "Инструмент поддержки решений для аналитиков кредитного риска.",
        "overview.total_applications": "Всего заявок",
        "overview.default_rate": "Доля дефолтов",
        "overview.model_auc": "ROC-AUC модели",
        "overview.threshold": "Текущий порог",
        "overview.project_goal": "Цель проекта",
        "overview.project_goal_text": (
            "Дашборд оценивает вероятность того, что клиент столкнется с трудностями при выплате кредита. "
            "Результат используется как поддержка аналитика, а не как автоматическое кредитное решение."
        ),
        "overview.recommendation_review": "Рекомендация: нужна проверка",
        "overview.data_scope": "Состав данных",
        "overview.data_scope_text": (
            "Признаки основной заявки объединены с историческими кредитными сигналами из датасета Home Credit."
        ),
        "overview.target_note": "TARGET: 0 означает выплату кредита, 1 означает трудности с выплатой.",
        "page.single.title": "Скоринг клиента",
        "page.single.subtitle": "Оценка вероятности дефолта по одной кредитной заявке.",
        "single.standard_profile": "Стандартный профиль",
        "single.stressed_profile": "Рискованный профиль",
        "single.client_profile": "Профиль клиента",
        "single.loan_parameters": "Параметры кредита",
        "single.external_history": "Внешние источники и кредитная история",
        "single.calculate": "Рассчитать риск",
        "field.age": "Возраст",
        "field.gender": "Пол",
        "field.contract_type": "Тип кредита",
        "field.education": "Образование",
        "field.family_status": "Семейное положение",
        "field.income_type": "Тип дохода",
        "field.occupation_type": "Профессия",
        "field.children_count": "Количество детей",
        "field.family_members": "Членов семьи",
        "field.total_income": "Доход",
        "field.credit_amount": "Сумма кредита",
        "field.annuity": "Платеж",
        "field.goods_price": "Стоимость товара",
        "field.employment_years": "Стаж, лет",
        "result.probability": "Вероятность дефолта",
        "result.waiting": "Ожидание",
        "result.waiting_hint": "Запустите скоринг, чтобы рассчитать риск.",
        "result.recommendation": "Рекомендация",
        "result.increasing": "Факторы повышения риска",
        "result.reducing": "Факторы снижения риска",
        "result.coverage": "Покрытие входных признаков модели",
        "result.missing_features": (
            "{count} исторических признаков не вводились вручную и были обработаны пайплайном модели."
        ),
        "result.extra_features": (
            "Эти поля показаны для контекста аналитика, но не используются текущей top-N моделью: {features}."
        ),
        "risk.low": "Низкий риск",
        "risk.medium": "Средний риск",
        "risk.high": "Высокий риск",
        "risk.unknown": "Неизвестный риск",
        "recommendation.low": "Приемлемый профиль риска",
        "recommendation.medium": "Требуется ручная проверка",
        "recommendation.high": "Заявка с высоким риском",
        "recommendation.unknown": "Требуется проверка",
        "page.batch.title": "Пакетный скоринг",
        "page.batch.subtitle": "Загрузка CSV и скоринг нескольких клиентов.",
        "batch.upload": "Загрузить CSV с заявками",
        "batch.callout": "Загрузите CSV с признаками Home Credit, чтобы запустить пакетный скоринг.",
        "batch.preview": "Предпросмотр",
        "batch.run": "Запустить пакетное предсказание",
        "batch.results": "Результаты скоринга",
        "batch.download": "Скачать CSV с предсказаниями",
        "batch.low": "Низкий",
        "batch.medium": "Средний",
        "batch.high": "Высокий",
        "page.data.title": "Анализ данных",
        "page.data.subtitle": "Качество датасета, баланс таргета и распределения признаков.",
        "data.target_distribution": "Распределение TARGET",
        "data.top_missing_values": "Топ пропусков",
        "data.credit_risk_correlations": "Корреляции с дефолтом",
        "data.pca_projection": "PCA-проекция",
        "data.missing_artifact": "Не найден артефакт: {path}",
        "data.missing_values_table": "Таблица пропусков",
        "page.explain.title": "Интерпретация модели",
        "page.explain.subtitle": "Глобальная важность признаков и локальные объяснения.",
        "explain.feature_unavailable": "Артефакт важности признаков недоступен.",
        "explain.global_importance": "Глобальная важность признаков",
        "page.performance.title": "Качество модели",
        "page.performance.subtitle": "Метрики экспериментов и панель анализа порога.",
        "performance.results_unavailable": "Результаты экспериментов недоступны.",
        "performance.experiment_table": "Таблица экспериментов",
        "performance.threshold_analysis": "Анализ порога",
        "performance.decision_threshold": "Порог решения",
        "performance.threshold_text": (
            "Сохраненная таблица экспериментов содержит агрегированные метрики на оценочном пороге модели. "
            "Слайдер оставлен для аналитического сценария; точные threshold-кривые требуют сохраненных "
            "вероятностей на validation."
        ),
        "performance.current_threshold": "Текущий порог",
    },
}

OPTION_TRANSLATIONS_RU = {
    "M": "Мужской",
    "F": "Женский",
    "XNA": "Не указано",
    "Cash loans": "Кредит наличными",
    "Revolving loans": "Возобновляемый кредит",
    "Secondary / secondary special": "Среднее / среднее специальное",
    "Higher education": "Высшее образование",
    "Incomplete higher": "Неполное высшее",
    "Lower secondary": "Неполное среднее",
    "Academic degree": "Ученая степень",
    "Single / not married": "Не женат / не замужем",
    "Married": "В браке",
    "Civil marriage": "Гражданский брак",
    "Separated": "Раздельно",
    "Widow": "Вдова / вдовец",
    "Working": "Работающий",
    "Commercial associate": "Коммерческий сотрудник",
    "Pensioner": "Пенсионер",
    "State servant": "Госслужащий",
    "Student": "Студент",
    "Laborers": "Рабочие",
    "Core staff": "Основной персонал",
    "Managers": "Менеджеры",
    "Sales staff": "Продажи",
    "Drivers": "Водители",
    "Accountants": "Бухгалтеры",
    "Other": "Другое",
}

DRIVER_TRANSLATIONS_RU = {
    "Low EXT_SOURCE_2 increases risk": "Низкий EXT_SOURCE_2 повышает риск",
    "Low EXT_SOURCE_3 increases risk": "Низкий EXT_SOURCE_3 повышает риск",
    "High credit / income ratio increases risk": "Высокое отношение кредита к доходу повышает риск",
    "High annuity / income ratio increases risk": "Высокое отношение платежа к доходу повышает риск",
    "Short employment history increases risk": "Короткая история занятости повышает риск",
    "Strong external score decreases risk": "Высокий внешний скоринг снижает риск",
    "Moderate annuity burden decreases risk": "Умеренная долговая нагрузка снижает риск",
    "Stable employment decreases risk": "Стабильная занятость снижает риск",
    "Stable family status decreases risk": "Стабильное семейное положение снижает риск",
    "No strong risk-increasing rule triggered": "Сильных факторов повышения риска не найдено",
    "No strong risk-reducing rule triggered": "Сильных факторов снижения риска не найдено",
}


def language_code_from_label(label: str) -> str:
    """Map a user-facing language label to a stable language code."""

    return LANGUAGE_CODES_BY_LABEL.get(label, "en")


def text(lang: str, key: str) -> str:
    """Return translated UI text with English and key fallbacks."""

    language = TEXT.get(lang, TEXT["en"])
    return language.get(key, TEXT["en"].get(key, key))


def page_options(lang: str) -> dict[str, str]:
    """Return localized sidebar labels mapped to stable page keys."""

    return {text(lang, f"nav.{page_key}"): page_key for page_key in PAGE_KEYS}


def translate_option(lang: str, option: object) -> str:
    """Translate categorical display values without changing model payload values."""

    value = str(option)
    if lang != "ru":
        return value
    return OPTION_TRANSLATIONS_RU.get(value, value)


def translate_driver(lang: str, driver: str) -> str:
    """Translate deterministic risk-driver text."""

    if lang != "ru":
        return driver
    return DRIVER_TRANSLATIONS_RU.get(driver, driver)


def localize_risk_tone(lang: str, risk_level: str, tone: Mapping[str, str]) -> dict[str, str]:
    """Return risk tone with localized label and unchanged colors."""

    normalized = risk_level.lower()
    key = f"risk.{normalized}"
    label = text(lang, key)
    if label == key:
        label = text(lang, "risk.unknown")
    return {**tone, "label": label}


def localized_recommendation(lang: str, risk_level: str) -> str:
    """Return localized analyst-facing recommendation."""

    normalized = risk_level.lower()
    key = f"recommendation.{normalized}"
    recommendation = text(lang, key)
    if recommendation == key:
        recommendation = text(lang, "recommendation.unknown")
    return recommendation
