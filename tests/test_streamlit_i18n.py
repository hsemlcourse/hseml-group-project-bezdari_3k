from src.ui.i18n import language_code_from_label, page_options, text, translate_option


def test_text_returns_russian_translation_and_english_fallback():
    assert text("ru", "page.overview.title") == "Home Credit Default Risk"
    assert text("ru", "sidebar.language") == "Язык"
    assert text("ru", "missing.translation.key") == "missing.translation.key"


def test_page_options_keep_stable_page_keys_across_languages():
    english = page_options("en")
    russian = page_options("ru")

    assert english["🏠 Overview"] == "overview"
    assert russian["🏠 Обзор"] == "overview"
    assert set(english.values()) == set(russian.values())


def test_language_code_from_label_maps_display_labels():
    assert language_code_from_label("English") == "en"
    assert language_code_from_label("Русский") == "ru"
    assert language_code_from_label("Unknown") == "en"


def test_translate_option_localizes_known_category_values():
    assert translate_option("ru", "Higher education") == "Высшее образование"
    assert translate_option("ru", "Cash loans") == "Кредит наличными"
    assert translate_option("en", "Higher education") == "Higher education"
    assert translate_option("ru", "Unmapped value") == "Unmapped value"
