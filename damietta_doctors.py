from __future__ import annotations

import html
import os
from pathlib import Path
from typing import Any

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from jinja2 import DictLoader, Environment, select_autoescape


# ============================================================
# CONFIGURATION
# ============================================================

APP_NAME = "دليل أطباء دمياط"
HOST = "127.0.0.1"
PORT = 8000

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "doctors.csv"

REQUIRED_COLUMNS = [
    "id",
    "name",
    "specialty",
    "area",
    "phone",
    "fee",
    "clinic_address",
    "description",
]

OPTIONAL_COLUMNS = [
    "image",
    "experience",
    "working_hours",
    "education",
    "verified",
]

PRIMARY_COLOR = "#0B84F3"
SECONDARY_COLOR = "#12B8A6"
DARK_COLOR = "#123047"
BACKGROUND_COLOR = "#F5F9FC"
WHITE_COLOR = "#FFFFFF"
GRAY_COLOR = "#6B7C8F"
GREEN_COLOR = "#22A06B"
ORANGE_COLOR = "#F4A340"
RED_COLOR = "#E45C5C"


# ============================================================
# CSV CACHE
# ============================================================

_doctors_cache: pd.DataFrame | None = None
_csv_last_modified: float | None = None

_csv_error: str | None = None


# ============================================================
# SAFE VALUE HELPERS
# ============================================================

def safe_text(value: Any, default: str = "") -> str:
    """
    Convert any value into safe displayable text.

    NaN / None / empty values become an empty string.
    """

    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass

    text = str(value).strip()

    if text.lower() in {
        "nan",
        "none",
        "null",
        "<na>",
    }:
        return default

    return text


def safe_fee(value: Any) -> str:
    """
    Format consultation fee without showing NaN.
    """

    text = safe_text(value)

    if not text:
        return "غير محدد"

    return text


def safe_phone(value: Any) -> str:
    """
    Keep phone number as text.
    """

    return safe_text(value)


def doctor_to_dict(row: pd.Series) -> dict[str, Any]:
    """
    Convert a Pandas row into a clean dictionary.

    Only values from the CSV are used.
    """

    result: dict[str, Any] = {}

    for column in row.index:
        result[column] = safe_text(row[column])

    return result


# ============================================================
# CSV LOADER
# ============================================================

def _read_csv_with_encoding(path: Path) -> pd.DataFrame:
    """
    Try UTF-8-SIG first, then UTF-8.
    """

    encodings = ["utf-8-sig", "utf-8"]

    last_error: Exception | None = None

    for encoding in encodings:
        try:
            return pd.read_csv(
                path,
                encoding=encoding,
                dtype=str,
                keep_default_na=False,
            )
        except UnicodeDecodeError as exc:
            last_error = exc

    if last_error:
        raise last_error

    raise ValueError("Unable to read CSV file.")


def _validate_columns(df: pd.DataFrame) -> None:
    """
    Validate required CSV columns.
    """

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        missing = ", ".join(missing_columns)

        raise ValueError(
            "ملف CSV ناقص. الأعمدة المطلوبة هي: "
            + ", ".join(REQUIRED_COLUMNS)
            + ". الأعمدة المفقودة: "
            + missing
        )


def load_doctors(force_reload: bool = False) -> pd.DataFrame:
    """
    Load doctors.csv.

    Uses modification time caching:
    - If CSV did not change, return cached DataFrame.
    - If CSV changed, reload automatically.
    - force_reload=True always reloads.
    """

    global _doctors_cache
    global _csv_last_modified
    global _csv_error

    if not CSV_PATH.exists():
        _doctors_cache = None
        _csv_last_modified = None

        _csv_error = (
            "ملف بيانات الأطباء غير موجود. "
            "تأكد من وجود doctors.csv بجانب damietta_doctors.py"
        )

        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    try:
        current_mtime = CSV_PATH.stat().st_mtime

        if (
            not force_reload
            and _doctors_cache is not None
            and _csv_last_modified == current_mtime
        ):
            return _doctors_cache

        df = _read_csv_with_encoding(CSV_PATH)

        _validate_columns(df)

        # Keep all values as strings.
        df = df.fillna("")

        # Normalize column names.
        df.columns = [
            safe_text(column)
            for column in df.columns
        ]

        # Clean every cell.
        for column in df.columns:
            df[column] = df[column].map(safe_text)

        _doctors_cache = df
        _csv_last_modified = current_mtime
        _csv_error = None

        return df

    except Exception as exc:
        _doctors_cache = None
        _csv_last_modified = None

        _csv_error = safe_text(
            str(exc),
            "حدث خطأ أثناء قراءة ملف بيانات الأطباء."
        )

        return pd.DataFrame(columns=REQUIRED_COLUMNS)


def get_csv_error() -> str | None:
    """
    Return current CSV error, if any.
    """

    # Make sure CSV state is current.
    load_doctors()

    return _csv_error


# ============================================================
# DATA HELPERS
# ============================================================

def get_unique_values(
    df: pd.DataFrame,
    column: str,
) -> list[str]:
    """
    Get unique non-empty values from a column.
    """

    if column not in df.columns:
        return []

    values = (
        df[column]
        .map(safe_text)
        .loc[lambda series: series != ""]
        .drop_duplicates()
        .tolist()
    )

    return sorted(values, key=lambda value: value.casefold())


def filter_doctors(
    df: pd.DataFrame,
    query: str = "",
    specialty: str = "",
    area: str = "",
) -> pd.DataFrame:
    """
    Server-side filtering.

    Search fields:
    - name
    - specialty
    - area

    All filters are applied together.
    """

    result = df.copy()

    query = safe_text(query)
    specialty = safe_text(specialty)
    area = safe_text(area)

    # التأكد من أن الأعمدة نصية وآمنة
    for column in ["name", "specialty", "area"]:
        if column not in result.columns:
            result[column] = ""

        result[column] = (
            result[column]
            .map(safe_text)
        )

    # البحث العام
    if query:
        query_lower = query.casefold()

        mask = (
            result["name"]
            .str.casefold()
            .str.contains(
                query_lower,
                regex=False,
                na=False,
            )
            |
            result["specialty"]
            .str.casefold()
            .str.contains(
                query_lower,
                regex=False,
                na=False,
            )
            |
            result["area"]
            .str.casefold()
            .str.contains(
                query_lower,
                regex=False,
                na=False,
            )
        )

        result = result.loc[mask]

    # فلترة التخصص
    if specialty:
        specialty_lower = specialty.casefold()

        result = result.loc[
            result["specialty"]
            .str.casefold()
            .eq(specialty_lower)
        ]

    # فلترة المنطقة
    if area:
        area_lower = area.casefold()

        result = result.loc[
            result["area"]
            .str.casefold()
            .eq(area_lower)
        ]

    return result


def get_statistics(df: pd.DataFrame) -> dict[str, int]:
    """
    Calculate statistics directly from CSV data.
    """

    return {
        "total_doctors": len(df),
        "total_specialties": (
            df["specialty"].replace("", pd.NA).dropna().nunique()
            if "specialty" in df.columns
            else 0
        ),
        "total_areas": (
            df["area"].replace("", pd.NA).dropna().nunique()
            if "area" in df.columns
            else 0
        ),
    }


def find_doctor_by_id(
    df: pd.DataFrame,
    doctor_id: str,
) -> dict[str, Any] | None:
    """
    Find doctor using the exact ID stored in CSV.
    """

    if "id" not in df.columns:
        return None

    doctor_id = safe_text(doctor_id)

    matches = df.loc[
        df["id"].map(safe_text) == doctor_id
    ]

    if matches.empty:
        return None

    return doctor_to_dict(matches.iloc[0])


def prepare_doctors(
    df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Convert DataFrame rows into dictionaries.
    """

    if df.empty:
        return []

    return [
        doctor_to_dict(row)
        for _, row in df.iterrows()
    ]


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title=APP_NAME,
    description="دليل أطباء دمياط",
    version="1.0.0",
)


# ============================================================
# JINJA2 TEMPLATES
# ============================================================

BASE_TEMPLATE = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">

    <meta
        name="viewport"
        content="width=device-width, initial-scale=1.0"
    >

    <meta
        name="description"
        content="دليل أطباء دمياط - ابحث عن الأطباء حسب التخصص والمنطقة"
    >

    <title>{{ title }} | دليل أطباء دمياط</title>

    <link
        rel="preconnect"
        href="https://fonts.googleapis.com"
    >

    <link
        rel="preconnect"
        href="https://fonts.gstatic.com"
        crossorigin
    >

    <link
        href="https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;600;700;800&display=swap"
        rel="stylesheet"
    >

    <link
        rel="stylesheet"
        href="/static/style.css"
    >
</head>

<body>

<header class="site-header">
    <div class="container header-inner">

        <a href="/" class="brand">
            <span class="brand-icon">✚</span>

            <span class="brand-text">
                <strong>دليل أطباء دمياط</strong>
                <small>دليلك للوصول إلى الطبيب المناسب</small>
            </span>
        </a>

        <nav class="main-nav">
            <a href="/">الرئيسية</a>
            <a href="/doctors">الأطباء</a>
        </nav>

    </div>
</header>

<main>
    {% block content %}{% endblock %}
</main>

<footer class="site-footer">

    <div class="container footer-grid">

        <div>
            <div class="footer-brand">
                <span class="brand-icon small">✚</span>
                <strong>دليل أطباء دمياط</strong>
            </div>

            <p>
                دليل إلكتروني بسيط يساعدك على استكشاف
                بيانات الأطباء في محافظة دمياط.
            </p>
        </div>

        <div>
            <h3>روابط سريعة</h3>

            <a href="/">الرئيسية</a>
            <a href="/doctors">جميع الأطباء</a>
        </div>

        <div>
            <h3>عن الموقع</h3>

            <p>
                الموقع مخصص لعرض بيانات الأطباء فقط،
                ولا يوفر خدمة الحجز أو المواعيد.
            </p>
        </div>

    </div>

    <div class="footer-bottom">
        <div class="container">
            © {{ current_year }} دليل أطباء دمياط
        </div>
    </div>

</footer>

<script src="/static/app.js"></script>

</body>
</html>
"""


HOME_TEMPLATE = """
{% extends "base.html" %}

{% block content %}

<section class="hero">

    <div class="container">

        <div class="hero-content">

            <div class="hero-badge">
                <span>●</span>
                دليل أطباء دمياط
            </div>

            <h1>
                ابحث عن طبيبك
                <span>في دمياط بسهولة</span>
            </h1>

            <p>
                استكشف الأطباء حسب التخصص والمنطقة
                واعرف بيانات التواصل وسعر الكشف.
            </p>

            <form
                action="/doctors"
                method="get"
                class="hero-search"
            >

                <div class="search-input-wrapper">
                    <span class="search-icon">⌕</span>

                    <input
                        type="search"
                        name="q"
                        placeholder="ابحث باسم الطبيب أو التخصص أو المنطقة..."
                        autocomplete="off"
                    >
                </div>

                <button
                    type="submit"
                    class="primary-button"
                >
                    بحث عن طبيب
                </button>

            </form>

            <div class="hero-note">
                بيانات الأطباء يتم تحميلها مباشرة من ملف الدليل.
            </div>

        </div>

        <div class="hero-visual">

            <div class="medical-circle">
                <div class="medical-cross">+</div>
            </div>

            <div class="floating-card floating-card-one">
                <span class="floating-icon">🩺</span>

                <div>
                    <strong>أطباء متخصصون</strong>
                    <small>في مختلف التخصصات</small>
                </div>
            </div>

            <div class="floating-card floating-card-two">
                <span class="floating-icon">📍</span>

                <div>
                    <strong>مناطق دمياط</strong>
                    <small>ابحث حسب منطقتك</small>
                </div>
            </div>

        </div>

    </div>

</section>


<section class="stats-section">

    <div class="container stats-grid">

        <div class="stat-card">
            <div class="stat-icon blue">👨‍⚕️</div>

            <div>
                <strong>{{ statistics.total_doctors }}</strong>
                <span>طبيب</span>
            </div>
        </div>

        <div class="stat-card">
            <div class="stat-icon teal">🩺</div>

            <div>
                <strong>{{ statistics.total_specialties }}</strong>
                <span>تخصص</span>
            </div>
        </div>

        <div class="stat-card">
            <div class="stat-icon orange">📍</div>

            <div>
                <strong>{{ statistics.total_areas }}</strong>
                <span>منطقة</span>
            </div>
        </div>

    </div>

</section>


<section class="section">

    <div class="container">

        <div class="section-heading">

            <div>
                <span class="section-kicker">EXPLORE</span>

                <h2>
                    استكشف التخصصات
                </h2>

                <p>
                    اختر التخصص للوصول إلى الأطباء المتاحين.
                </p>
            </div>

            <a
                href="/doctors"
                class="text-link"
            >
                عرض جميع الأطباء ←
            </a>

        </div>


        {% if specialties %}

        <div class="specialties-grid">

            {% for specialty in specialties %}

            <a
                class="specialty-card"
                href="/doctors?specialty={{ specialty | urlencode }}"
            >

                <span class="specialty-icon">
                    ✚
                </span>

                <span class="specialty-content">
                    <strong>{{ specialty }}</strong>
                    <small>عرض الأطباء</small>
                </span>

                <span class="specialty-arrow">
                    ←
                </span>

            </a>

            {% endfor %}

        </div>

        {% else %}

        <div class="empty-state">
            <div class="empty-icon">🩺</div>

            <h3>لا توجد تخصصات حاليًا</h3>

            <p>
                أضف بيانات الأطباء إلى doctors.csv
                لعرض التخصصات هنا.
            </p>
        </div>

        {% endif %}

    </div>

</section>


<section class="section section-soft">

    <div class="container">

        <div class="section-heading centered">

            <div>
                <span class="section-kicker">DOCTORS</span>

                <h2>
                    الأطباء
                </h2>

                <p>
                    بعض الأطباء الموجودين في دليل دمياط.
                </p>
            </div>

        </div>


        {% if featured_doctors %}

        <div class="doctors-grid">

            {% for doctor in featured_doctors %}

            {% include "doctor_card.html" %}

            {% endfor %}

        </div>

        {% else %}

        <div class="empty-state">
            <div class="empty-icon">👨‍⚕️</div>

            <h3>لا توجد بيانات أطباء</h3>

            <p>
                تأكد من وجود doctors.csv وإضافة بيانات صحيحة إليه.
            </p>
        </div>

        {% endif %}


        {% if statistics.total_doctors > 6 %}

        <div class="center-button">
            <a
                href="/doctors"
                class="primary-button"
            >
                استكشف جميع الأطباء
            </a>
        </div>

        {% endif %}

    </div>

</section>


<section class="section">

    <div class="container">

        <div class="section-heading centered">

            <div>
                <span class="section-kicker">HOW IT WORKS</span>

                <h2>
                    كيف تستخدم الموقع؟
                </h2>

                <p>
                    أربع خطوات بسيطة للوصول إلى بيانات الطبيب.
                </p>
            </div>

        </div>


        <div class="steps-grid">

            <div class="step-card">
                <span class="step-number">01</span>
                <div class="step-icon">🩺</div>
                <h3>اختر التخصص</h3>
                <p>
                    حدد التخصص الطبي الذي تبحث عنه.
                </p>
            </div>

            <div class="step-card">
                <span class="step-number">02</span>
                <div class="step-icon">📍</div>
                <h3>حدد المنطقة</h3>
                <p>
                    استخدم المنطقة لتضييق نتائج البحث.
                </p>
            </div>

            <div class="step-card">
                <span class="step-number">03</span>
                <div class="step-icon">👨‍⚕️</div>
                <h3>اختر الطبيب</h3>
                <p>
                    استعرض البطاقات واختر الطبيب المطلوب.
                </p>
            </div>

            <div class="step-card">
                <span class="step-number">04</span>
                <div class="step-icon">📞</div>
                <h3>شاهد بيانات التواصل</h3>
                <p>
                    افتح ملف الطبيب لمعرفة التفاصيل المتاحة.
                </p>
            </div>

        </div>

    </div>

</section>

{% endblock %}
"""


DOCTOR_CARD_TEMPLATE = """
<article class="doctor-card">

    <a
        href="/doctors/{{ doctor.id }}"
        class="doctor-card-link"
        aria-label="عرض ملف {{ doctor.name }}"
    >

        <div class="doctor-card-top">

            <div class="doctor-avatar">

                {% if doctor.image %}

                    <img
                        src="{{ doctor.image }}"
                        alt="{{ doctor.name }}"
                        loading="lazy"
                    >

                {% else %}

                    <span>👨‍⚕️</span>

                {% endif %}

            </div>


            {% if doctor.verified %}

            <span class="verified-badge">
                ✓ موثق
            </span>

            {% endif %}

        </div>


        <div class="doctor-card-body">

            <h3>
                {{ doctor.name or "طبيب" }}
            </h3>

            {% if doctor.specialty %}

            <span class="doctor-specialty">
                {{ doctor.specialty }}
            </span>

            {% endif %}


            <div class="doctor-info-list">

                {% if doctor.area %}

                <div class="doctor-info-item">
                    <span>📍</span>
                    <span>{{ doctor.area }}</span>
                </div>

                {% endif %}


                {% if doctor.phone %}

                <div class="doctor-info-item">
                    <span>📞</span>
                    <span dir="ltr">{{ doctor.phone }}</span>
                </div>

                {% endif %}


                {% if doctor.fee %}

                <div class="doctor-info-item">
                    <span>💰</span>
                    <span>{{ doctor.fee }} جنيه</span>
                </div>

                {% endif %}

            </div>

        </div>


        <div class="doctor-card-footer">

            <span>
                عرض الملف
            </span>

            <span class="arrow">
                ←
            </span>

        </div>

    </a>

</article>
"""


DOCTORS_TEMPLATE = """
{% extends "base.html" %}

{% block content %}

<section class="page-hero">

    <div class="container">

        <div class="breadcrumb">
            <a href="/">الرئيسية</a>
            <span>←</span>
            <span>الأطباء</span>
        </div>

        <div class="page-hero-content">

            <div>
                <span class="section-kicker">DIRECTORY</span>

                <h1>
                    دليل الأطباء
                </h1>

                <p>
                    ابحث واستكشف الأطباء حسب الاسم والتخصص والمنطقة.
                </p>
            </div>

            <div class="result-counter">
                <strong>{{ result_count }}</strong>
                <span>نتيجة</span>
            </div>

        </div>

    </div>

</section>


<section class="directory-section">

    <div class="container">

        <form
            action="/doctors"
            method="get"
            class="filters-card"
        >

            <div class="filter-field search-field">

                <label for="q">
                    البحث
                </label>

                <div class="input-with-icon">

                    <span>⌕</span>

                    <input
                        id="q"
                        type="search"
                        name="q"
                        value="{{ q }}"
                        placeholder="اسم الطبيب، التخصص، المنطقة..."
                    >

                </div>

            </div>


            <div class="filter-field">

                <label for="specialty">
                    التخصص
                </label>

                <select
                    id="specialty"
                    name="specialty"
                >

                    <option value="">
                        كل التخصصات
                    </option>

                    {% for item in specialties %}

                    <option
                        value="{{ item }}"
                        {% if item == specialty %}selected{% endif %}
                    >
                        {{ item }}
                    </option>

                    {% endfor %}

                </select>

            </div>


            <div class="filter-field">

                <label for="area">
                    المنطقة
                </label>

                <select
                    id="area"
                    name="area"
                >

                    <option value="">
                        كل المناطق
                    </option>

                    {% for item in areas %}

                    <option
                        value="{{ item }}"
                        {% if item == area %}selected{% endif %}
                    >
                        {{ item }}
                    </option>

                    {% endfor %}

                </select>

            </div>


            <div class="filter-actions">

                <button
                    type="submit"
                    class="primary-button"
                >
                    بحث
                </button>

                <a
                    href="/doctors"
                    class="secondary-button"
                >
                    إعادة ضبط
                </a>

            </div>

        </form>


        {% if csv_error %}

        <div class="alert alert-error">

            <div class="alert-icon">!</div>

            <div>
                <strong>تعذر تحميل بيانات الأطباء</strong>

                <p>
                    {{ csv_error }}
                </p>
            </div>

        </div>

        {% endif %}


        {% if doctors %}

        <div class="directory-toolbar">

            <span>
                تم العثور على
                <strong>{{ result_count }}</strong>
                طبيب
            </span>

            {% if q or specialty or area %}

            <span class="active-filter">
                توجد فلاتر مفعلة
            </span>

            {% endif %}

        </div>


        <div class="doctors-grid">

            {% for doctor in doctors %}

            {% include "doctor_card.html" %}

            {% endfor %}

        </div>


        {% else %}

        <div class="empty-state large">

            <div class="empty-icon">
                🔎
            </div>

            <h2>
                لم نجد أطباء يطابقون بحثك
            </h2>

            <p>
                جرّب تغيير التخصص أو المنطقة أو كلمة البحث.
            </p>

            <a
                href="/doctors"
                class="primary-button"
            >
                عرض جميع الأطباء
            </a>

        </div>

        {% endif %}

    </div>

</section>

{% endblock %}
"""


PROFILE_TEMPLATE = """
{% extends "base.html" %}

{% block content %}

<section class="profile-page">

    <div class="container">

        <div class="breadcrumb">
            <a href="/">الرئيسية</a>
            <span>←</span>
            <a href="/doctors">الأطباء</a>
            <span>←</span>
            <span>ملف الطبيب</span>
        </div>


        <div class="profile-layout">

            <aside class="profile-sidebar">

                <div class="profile-avatar">

                    {% if doctor.image %}

                    <img
                        src="{{ doctor.image }}"
                        alt="{{ doctor.name }}"
                    >

                    {% else %}

                    <span>👨‍⚕️</span>

                    {% endif %}

                </div>


                {% if doctor.verified %}

                <div class="profile-verified">
                    ✓ طبيب موثق
                </div>

                {% endif %}


                <a
                    href="tel:{{ doctor.phone }}"
                    class="call-button {% if not doctor.phone %}disabled{% endif %}"
                    {% if not doctor.phone %}
                    aria-disabled="true"
                    onclick="return false;"
                    {% endif %}
                >
                    <span>📞</span>
                    اتصال بالطبيب
                </a>


                <a
                    href="/doctors"
                    class="secondary-button full-width"
                >
                    ← العودة إلى الأطباء
                </a>

            </aside>


            <div class="profile-main">

                <div class="profile-header">

                    <div>

                        <span class="profile-specialty">
                            {{ doctor.specialty or "تخصص غير محدد" }}
                        </span>

                        <h1>
                            {{ doctor.name or "طبيب" }}
                        </h1>

                        {% if doctor.area %}

                        <div class="profile-location">
                            📍 {{ doctor.area }}
                        </div>

                        {% endif %}

                    </div>

                </div>


                <div class="profile-highlights">

                    <div class="highlight-card">

                        <span class="highlight-icon">💰</span>

                        <div>
                            <small>سعر الكشف</small>

                            <strong>
                                {% if doctor.fee %}
                                    {{ doctor.fee }} جنيه
                                {% else %}
                                    غير محدد
                                {% endif %}
                            </strong>
                        </div>

                    </div>


                    <div class="highlight-card">

                        <span class="highlight-icon">📞</span>

                        <div>
                            <small>رقم الهاتف</small>

                            <strong dir="ltr">
                                {{ doctor.phone or "غير متاح" }}
                            </strong>
                        </div>

                    </div>


                    <div class="highlight-card">

                        <span class="highlight-icon">📍</span>

                        <div>
                            <small>المنطقة</small>

                            <strong>
                                {{ doctor.area or "غير محددة" }}
                            </strong>
                        </div>

                    </div>

                </div>


                <section class="profile-section">

                    <h2>
                        معلومات الطبيب
                    </h2>

                    <div class="details-grid">

                        {% if doctor.clinic_address %}

                        <div class="detail-row">

                            <span class="detail-icon">🏥</span>

                            <div>
                                <small>عنوان العيادة</small>
                                <strong>
                                    {{ doctor.clinic_address }}
                                </strong>
                            </div>

                        </div>

                        {% endif %}


                        {% if doctor.experience %}

                        <div class="detail-row">

                            <span class="detail-icon">⭐</span>

                            <div>
                                <small>الخبرة</small>
                                <strong>
                                    {{ doctor.experience }}
                                </strong>
                            </div>

                        </div>

                        {% endif %}


                        {% if doctor.working_hours %}

                        <div class="detail-row">

                            <span class="detail-icon">🕐</span>

                            <div>
                                <small>مواعيد العمل</small>
                                <strong>
                                    {{ doctor.working_hours }}
                                </strong>
                            </div>

                        </div>

                        {% endif %}


                        {% if doctor.education %}

                        <div class="detail-row">

                            <span class="detail-icon">🎓</span>

                            <div>
                                <small>التعليم</small>
                                <strong>
                                    {{ doctor.education }}
                                </strong>
                            </div>

                        </div>

                        {% endif %}

                    </div>

                </section>


                {% if doctor.description %}

                <section class="profile-section">

                    <h2>
                        نبذة عن الطبيب
                    </h2>

                    <div class="description-box">
                        {{ doctor.description }}
                    </div>

                </section>

                {% endif %}


                {% set known_columns = [
                    "id",
                    "name",
                    "specialty",
                    "area",
                    "phone",
                    "fee",
                    "clinic_address",
                    "description",
                    "image",
                    "experience",
                    "working_hours",
                    "education",
                    "verified"
                ] %}

                {% set extra_fields = [] %}

                {% for key, value in doctor.items() %}

                    {% if key not in known_columns and value %}

                        {% set _ = extra_fields.append((key, value)) %}

                    {% endif %}

                {% endfor %}


                {% if extra_fields %}

                <section class="profile-section">

                    <h2>
                        معلومات إضافية
                    </h2>

                    <div class="details-grid">

                        {% for key, value in extra_fields %}

                        <div class="detail-row">

                            <span class="detail-icon">
                                •
                            </span>

                            <div>
                                <small>{{ key }}</small>
                                <strong>{{ value }}</strong>
                            </div>

                        </div>

                        {% endfor %}

                    </div>

                </section>

                {% endif %}


                <div class="profile-disclaimer">

                    <span>ⓘ</span>

                    <p>
                        المعلومات المعروضة في هذا الملف مأخوذة من
                        ملف بيانات الدليل. لا يوفر الموقع خدمة الحجز
                        أو تأكيد المواعيد.
                    </p>

                </div>

            </div>

        </div>

    </div>

</section>

{% endblock %}
"""


ERROR_TEMPLATE = """
{% extends "base.html" %}

{% block content %}

<section class="error-page">

    <div class="container">

        <div class="error-card">

            <div class="error-number">
                {{ error_code }}
            </div>

            <div class="error-icon">
                {{ icon }}
            </div>

            <h1>
                {{ title }}
            </h1>

            <p>
                {{ message }}
            </p>

            <a
                href="{{ button_url }}"
                class="primary-button"
            >
                {{ button_text }}
            </a>

        </div>

    </div>

</section>

{% endblock %}
"""


# ============================================================
# CSS
# ============================================================

STYLE_CSS = r"""
:root {
    --primary: #0B84F3;
    --primary-dark: #0872d4;
    --secondary: #12B8A6;
    --dark: #123047;
    --background: #F5F9FC;
    --white: #FFFFFF;
    --gray: #6B7C8F;
    --gray-light: #E7EEF4;
    --green: #22A06B;
    --orange: #F4A340;
    --red: #E45C5C;
    --shadow: 0 12px 35px rgba(18, 48, 71, 0.08);
    --shadow-hover: 0 18px 45px rgba(18, 48, 71, 0.13);
    --radius: 18px;
    --radius-small: 12px;
}

* {
    box-sizing: border-box;
}

html {
    scroll-behavior: smooth;
}

body {
    margin: 0;
    background: var(--background);
    color: var(--dark);
    font-family: "Cairo", Arial, sans-serif;
    line-height: 1.7;
}

a {
    color: inherit;
    text-decoration: none;
}

button,
input,
select {
    font-family: inherit;
}

button {
    cursor: pointer;
}

img {
    max-width: 100%;
    display: block;
}

.container {
    width: min(1180px, calc(100% - 40px));
    margin-inline: auto;
}


/* ============================================================
   HEADER
   ============================================================ */

.site-header {
    background: rgba(255, 255, 255, 0.94);
    backdrop-filter: blur(16px);
    border-bottom: 1px solid rgba(18, 48, 71, 0.07);
    position: sticky;
    top: 0;
    z-index: 100;
}

.header-inner {
    min-height: 76px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 30px;
}

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
}

.brand-icon {
    width: 46px;
    height: 46px;
    display: grid;
    place-items: center;
    border-radius: 14px;
    color: white;
    background: linear-gradient(
        135deg,
        var(--primary),
        var(--secondary)
    );
    font-size: 25px;
    font-weight: 800;
    box-shadow: 0 8px 22px rgba(11, 132, 243, 0.23);
}

.brand-icon.small {
    width: 38px;
    height: 38px;
    font-size: 20px;
    border-radius: 11px;
}

.brand-text {
    display: flex;
    flex-direction: column;
}

.brand-text strong {
    font-size: 16px;
    line-height: 1.3;
}

.brand-text small {
    color: var(--gray);
    font-size: 10px;
}

.main-nav {
    display: flex;
    align-items: center;
    gap: 30px;
}

.main-nav a {
    color: var(--gray);
    font-size: 14px;
    font-weight: 700;
    transition: 0.2s ease;
}

.main-nav a:hover {
    color: var(--primary);
}


/* ============================================================
   HERO
   ============================================================ */

.hero {
    overflow: hidden;
    position: relative;
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(18, 184, 166, 0.12),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 20%,
            rgba(11, 132, 243, 0.12),
            transparent 35%
        ),
        var(--white);
    padding: 80px 0 100px;
}

.hero::after {
    content: "";
    position: absolute;
    width: 500px;
    height: 500px;
    border-radius: 50%;
    border: 1px solid rgba(11, 132, 243, 0.08);
    left: -250px;
    bottom: -350px;
}

.hero .container {
    min-height: 440px;
    display: grid;
    grid-template-columns: 1.1fr 0.9fr;
    align-items: center;
    gap: 60px;
}

.hero-content {
    position: relative;
    z-index: 2;
}

.hero-badge {
    width: fit-content;
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 7px 13px;
    border-radius: 999px;
    background: rgba(11, 132, 243, 0.08);
    color: var(--primary);
    font-size: 12px;
    font-weight: 800;
    margin-bottom: 20px;
}

.hero-badge span {
    font-size: 9px;
}

.hero h1 {
    margin: 0;
    max-width: 700px;
    font-size: clamp(36px, 5vw, 62px);
    line-height: 1.2;
    letter-spacing: -1.8px;
}

.hero h1 span {
    color: var(--primary);
    display: block;
}

.hero p {
    color: var(--gray);
    font-size: 17px;
    max-width: 620px;
    margin: 20px 0 28px;
}

.hero-search {
    max-width: 720px;
    background: var(--white);
    border: 1px solid var(--gray-light);
    border-radius: 16px;
    padding: 7px;
    display: flex;
    align-items: center;
    gap: 8px;
    box-shadow: var(--shadow);
}

.search-input-wrapper {
    flex: 1;
    position: relative;
    display: flex;
    align-items: center;
}

.search-input-wrapper .search-icon {
    position: absolute;
    right: 14px;
    font-size: 23px;
    color: var(--gray);
}

.search-input-wrapper input {
    width: 100%;
    border: 0;
    outline: 0;
    padding: 14px 46px 14px 14px;
    color: var(--dark);
    font-size: 14px;
    background: transparent;
}

.primary-button {
    border: 0;
    border-radius: 12px;
    background: var(--primary);
    color: white;
    padding: 13px 21px;
    font-size: 13px;
    font-weight: 800;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    transition: 0.2s ease;
}

.primary-button:hover {
    background: var(--primary-dark);
    transform: translateY(-1px);
}

.hero-note {
    color: var(--gray);
    font-size: 11px;
    margin-top: 11px;
}

.hero-visual {
    position: relative;
    min-height: 400px;
    display: grid;
    place-items: center;
}

.medical-circle {
    width: min(340px, 75vw);
    aspect-ratio: 1;
    border-radius: 50%;
    background:
        radial-gradient(
            circle,
            rgba(11, 132, 243, 0.14),
            rgba(18, 184, 166, 0.08) 50%,
            transparent 70%
        );
    border: 1px solid rgba(11, 132, 243, 0.12);
    display: grid;
    place-items: center;
}

.medical-cross {
    width: 130px;
    height: 130px;
    display: grid;
    place-items: center;
    border-radius: 38px;
    background: var(--white);
    color: var(--primary);
    font-size: 100px;
    font-weight: 300;
    line-height: 1;
    box-shadow: var(--shadow);
}

.floating-card {
    position: absolute;
    background: rgba(255, 255, 255, 0.96);
    border: 1px solid rgba(18, 48, 71, 0.08);
    border-radius: 16px;
    box-shadow: var(--shadow);
    padding: 13px 15px;
    display: flex;
    align-items: center;
    gap: 10px;
    min-width: 205px;
}

.floating-card strong,
.floating-card small {
    display: block;
}

.floating-card strong {
    font-size: 12px;
}

.floating-card small {
    color: var(--gray);
    font-size: 9px;
}

.floating-icon {
    width: 38px;
    height: 38px;
    border-radius: 11px;
    display: grid;
    place-items: center;
    background: rgba(11, 132, 243, 0.08);
}

.floating-card-one {
    top: 35px;
    right: 0;
}

.floating-card-two {
    bottom: 35px;
    left: 0;
}


/* ============================================================
   STATS
   ============================================================ */

.stats-section {
    margin-top: -35px;
    position: relative;
    z-index: 5;
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 18px;
}

.stat-card {
    background: var(--white);
    border: 1px solid rgba(18, 48, 71, 0.06);
    border-radius: var(--radius);
    padding: 23px;
    box-shadow: var(--shadow);
    display: flex;
    align-items: center;
    gap: 16px;
}

.stat-icon {
    width: 55px;
    height: 55px;
    border-radius: 15px;
    display: grid;
    place-items: center;
    font-size: 24px;
}

.stat-icon.blue {
    background: rgba(11, 132, 243, 0.1);
}

.stat-icon.teal {
    background: rgba(18, 184, 166, 0.1);
}

.stat-icon.orange {
    background: rgba(244, 163, 64, 0.13);
}

.stat-card strong {
    display: block;
    font-size: 25px;
    line-height: 1.2;
}

.stat-card span {
    color: var(--gray);
    font-size: 12px;
}


/* ============================================================
   SECTIONS
   ============================================================ */

.section {
    padding: 90px 0;
}

.section-soft {
    background: #eef6fb;
}

.section-heading {
    display: flex;
    align-items: end;
    justify-content: space-between;
    gap: 30px;
    margin-bottom: 35px;
}

.section-heading.centered {
    justify-content: center;
    text-align: center;
}

.section-kicker {
    display: block;
    color: var(--primary);
    font-size: 10px;
    letter-spacing: 1.5px;
    font-weight: 800;
    margin-bottom: 7px;
}

.section-heading h2 {
    margin: 0;
    font-size: 31px;
    line-height: 1.3;
}

.section-heading p {
    color: var(--gray);
    margin: 8px 0 0;
    font-size: 13px;
}

.text-link {
    color: var(--primary);
    font-size: 13px;
    font-weight: 800;
}


/* ============================================================
   SPECIALTIES
   ============================================================ */

.specialties-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 15px;
}

.specialty-card {
    background: var(--white);
    border: 1px solid rgba(18, 48, 71, 0.06);
    border-radius: 16px;
    padding: 18px;
    display: flex;
    align-items: center;
    gap: 13px;
    transition: 0.25s ease;
}

.specialty-card:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-hover);
    border-color: rgba(11, 132, 243, 0.15);
}

.specialty-icon {
    width: 44px;
    height: 44px;
    border-radius: 13px;
    display: grid;
    place-items: center;
    color: var(--primary);
    background: rgba(11, 132, 243, 0.08);
    font-size: 22px;
}

.specialty-content {
    flex: 1;
}

.specialty-content strong,
.specialty-content small {
    display: block;
}

.specialty-content strong {
    font-size: 13px;
}

.specialty-content small {
    color: var(--gray);
    font-size: 9px;
}

.specialty-arrow {
    color: var(--gray);
}


/* ============================================================
   DOCTORS GRID
   ============================================================ */

.doctors-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
}

.doctor-card {
    background: var(--white);
    border-radius: var(--radius);
    overflow: hidden;
    border: 1px solid rgba(18, 48, 71, 0.06);
    box-shadow: 0 5px 22px rgba(18, 48, 71, 0.045);
    transition: 0.25s ease;
}

.doctor-card:hover {
    transform: translateY(-5px);
    box-shadow: var(--shadow-hover);
}

.doctor-card-link {
    display: block;
}

.doctor-card-top {
    min-height: 125px;
    padding: 18px;
    background:
        linear-gradient(
            135deg,
            rgba(11, 132, 243, 0.1),
            rgba(18, 184, 166, 0.08)
        );
    position: relative;
    display: flex;
    align-items: center;
}

.doctor-avatar {
    width: 78px;
    height: 78px;
    border-radius: 24px;
    overflow: hidden;
    background: var(--white);
    border: 4px solid rgba(255, 255, 255, 0.8);
    display: grid;
    place-items: center;
    font-size: 40px;
    box-shadow: 0 8px 20px rgba(18, 48, 71, 0.08);
}

.doctor-avatar img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

.verified-badge {
    position: absolute;
    top: 16px;
    left: 16px;
    background: rgba(34, 160, 107, 0.1);
    color: var(--green);
    border-radius: 999px;
    padding: 4px 9px;
    font-size: 9px;
    font-weight: 800;
}

.doctor-card-body {
    padding: 20px;
}

.doctor-card-body h3 {
    margin: 0 0 5px;
    font-size: 17px;
}

.doctor-specialty {
    display: inline-block;
    color: var(--primary);
    background: rgba(11, 132, 243, 0.08);
    border-radius: 999px;
    padding: 3px 10px;
    font-size: 10px;
    font-weight: 800;
}

.doctor-info-list {
    display: grid;
    gap: 9px;
    margin-top: 17px;
}

.doctor-info-item {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--gray);
    font-size: 11px;
}

.doctor-card-footer {
    border-top: 1px solid var(--gray-light);
    padding: 13px 20px;
    color: var(--primary);
    font-size: 11px;
    font-weight: 800;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.doctor-card-footer .arrow {
    font-size: 17px;
}

.center-button {
    text-align: center;
    margin-top: 35px;
}


/* ============================================================
   STEPS
   ============================================================ */

.steps-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 18px;
}

.step-card {
    background: var(--white);
    border: 1px solid rgba(18, 48, 71, 0.06);
    border-radius: var(--radius);
    padding: 24px;
    position: relative;
    overflow: hidden;
}

.step-number {
    position: absolute;
    top: 10px;
    left: 14px;
    color: rgba(11, 132, 243, 0.08);
    font-size: 48px;
    font-weight: 800;
    line-height: 1;
}

.step-icon {
    width: 50px;
    height: 50px;
    border-radius: 15px;
    display: grid;
    place-items: center;
    background: rgba(11, 132, 243, 0.08);
    font-size: 23px;
    margin-bottom: 18px;
}

.step-card h3 {
    margin: 0 0 7px;
    font-size: 15px;
}

.step-card p {
    margin: 0;
    color: var(--gray);
    font-size: 11px;
}


/* ============================================================
   PAGE HERO
   ============================================================ */

.page-hero {
    background: var(--white);
    border-bottom: 1px solid rgba(18, 48, 71, 0.06);
    padding: 35px 0 45px;
}

.breadcrumb {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    color: var(--gray);
    font-size: 10px;
    margin-bottom: 28px;
}

.breadcrumb a:hover {
    color: var(--primary);
}

.page-hero-content {
    display: flex;
    align-items: end;
    justify-content: space-between;
    gap: 30px;
}

.page-hero h1 {
    margin: 0;
    font-size: 40px;
}

.page-hero p {
    color: var(--gray);
    margin: 7px 0 0;
    font-size: 13px;
}

.result-counter {
    min-width: 110px;
    padding: 13px 18px;
    background: rgba(11, 132, 243, 0.07);
    border-radius: 15px;
    text-align: center;
}

.result-counter strong {
    display: block;
    color: var(--primary);
    font-size: 24px;
    line-height: 1.2;
}

.result-counter span {
    color: var(--gray);
    font-size: 9px;
}


/* ============================================================
   DIRECTORY
   ============================================================ */

.directory-section {
    padding: 45px 0 90px;
}

.filters-card {
    background: var(--white);
    border: 1px solid rgba(18, 48, 71, 0.06);
    box-shadow: var(--shadow);
    border-radius: var(--radius);
    padding: 20px;
    display: grid;
    grid-template-columns: 1.5fr 1fr 1fr auto;
    gap: 14px;
    align-items: end;
    margin-bottom: 28px;
}

.filter-field label {
    display: block;
    font-size: 10px;
    font-weight: 800;
    margin-bottom: 7px;
}

.filter-field input,
.filter-field select {
    width: 100%;
    height: 48px;
    border: 1px solid var(--gray-light);
    background: var(--background);
    color: var(--dark);
    border-radius: 11px;
    outline: none;
    padding: 0 13px;
    font-size: 12px;
}

.filter-field input:focus,
.filter-field select:focus {
    border-color: rgba(11, 132, 243, 0.5);
    box-shadow: 0 0 0 3px rgba(11, 132, 243, 0.08);
}

.input-with-icon {
    position: relative;
}

.input-with-icon span {
    position: absolute;
    right: 13px;
    top: 8px;
    color: var(--gray);
    font-size: 22px;
}

.input-with-icon input {
    padding-right: 42px;
}

.filter-actions {
    display: flex;
    gap: 7px;
}

.secondary-button {
    height: 48px;
    border: 1px solid var(--gray-light);
    border-radius: 11px;
    padding: 0 17px;
    background: var(--white);
    color: var(--dark);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    font-weight: 800;
    transition: 0.2s ease;
}

.secondary-button:hover {
    border-color: var(--primary);
    color: var(--primary);
}

.filter-actions .primary-button {
    height: 48px;
}

.directory-toolbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
    margin-bottom: 20px;
    color: var(--gray);
    font-size: 11px;
}

.directory-toolbar strong {
    color: var(--dark);
}

.active-filter {
    color: var(--primary);
    background: rgba(11, 132, 243, 0.08);
    padding: 5px 10px;
    border-radius: 999px;
}


/* ============================================================
   PROFILE
   ============================================================ */

.profile-page {
    padding: 35px 0 90px;
}

.profile-layout {
    display: grid;
    grid-template-columns: 290px 1fr;
    gap: 25px;
    align-items: start;
}

.profile-sidebar {
    background: var(--white);
    border-radius: var(--radius);
    border: 1px solid rgba(18, 48, 71, 0.06);
    box-shadow: var(--shadow);
    padding: 25px;
    text-align: center;
    position: sticky;
    top: 100px;
}

.profile-avatar {
    width: 145px;
    height: 145px;
    border-radius: 38px;
    margin: 0 auto 15px;
    background:
        linear-gradient(
            135deg,
            rgba(11, 132, 243, 0.12),
            rgba(18, 184, 166, 0.1)
        );
    display: grid;
    place-items: center;
    font-size: 70px;
    overflow: hidden;
}

.profile-avatar img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

.profile-verified {
    display: inline-block;
    color: var(--green);
    background: rgba(34, 160, 107, 0.1);
    padding: 5px 12px;
    border-radius: 999px;
    font-size: 10px;
    font-weight: 800;
    margin-bottom: 20px;
}

.call-button {
    min-height: 50px;
    width: 100%;
    border-radius: 12px;
    background: var(--green);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 9px;
    font-size: 12px;
    font-weight: 800;
    margin-bottom: 10px;
    transition: 0.2s ease;
}

.call-button:hover {
    filter: brightness(0.95);
    transform: translateY(-1px);
}

.call-button.disabled {
    background: #b7c3cc;
    cursor: not-allowed;
}

.full-width {
    width: 100%;
}

.profile-main {
    min-width: 0;
}

.profile-header {
    background: var(--white);
    border-radius: var(--radius);
    border: 1px solid rgba(18, 48, 71, 0.06);
    box-shadow: var(--shadow);
    padding: 32px;
}

.profile-specialty {
    display: inline-block;
    background: rgba(11, 132, 243, 0.08);
    color: var(--primary);
    border-radius: 999px;
    padding: 5px 12px;
    font-size: 10px;
    font-weight: 800;
}

.profile-header h1 {
    font-size: 34px;
    margin: 12px 0 3px;
}

.profile-location {
    color: var(--gray);
    font-size: 12px;
}

.profile-highlights {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin-top: 15px;
}

.highlight-card {
    background: var(--white);
    border: 1px solid rgba(18, 48, 71, 0.06);
    border-radius: 15px;
    padding: 17px;
    display: flex;
    align-items: center;
    gap: 11px;
}

.highlight-icon {
    width: 42px;
    height: 42px;
    display: grid;
    place-items: center;
    border-radius: 12px;
    background: rgba(11, 132, 243, 0.08);
}

.highlight-card small,
.highlight-card strong {
    display: block;
}

.highlight-card small {
    color: var(--gray);
    font-size: 9px;
}

.highlight-card strong {
    font-size: 12px;
    margin-top: 2px;
}

.profile-section {
    background: var(--white);
    border: 1px solid rgba(18, 48, 71, 0.06);
    border-radius: var(--radius);
    box-shadow: 0 5px 20px rgba(18, 48, 71, 0.035);
    padding: 26px;
    margin-top: 15px;
}

.profile-section h2 {
    font-size: 18px;
    margin: 0 0 20px;
}

.details-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 13px;
}

.detail-row {
    display: flex;
    gap: 12px;
    padding: 15px;
    border-radius: 13px;
    background: var(--background);
}

.detail-icon {
    width: 38px;
    height: 38px;
    flex-shrink: 0;
    display: grid;
    place-items: center;
    background: var(--white);
    border-radius: 10px;
}

.detail-row small,
.detail-row strong {
    display: block;
}

.detail-row small {
    color: var(--gray);
    font-size: 9px;
}

.detail-row strong {
    font-size: 11px;
    margin-top: 3px;
    word-break: break-word;
}

.description-box {
    color: var(--gray);
    background: var(--background);
    border-radius: 14px;
    padding: 18px;
    font-size: 12px;
    white-space: pre-line;
}

.profile-disclaimer {
    display: flex;
    gap: 10px;
    margin-top: 15px;
    padding: 15px;
    border-radius: 14px;
    background: rgba(244, 163, 64, 0.1);
    color: #825514;
}

.profile-disclaimer span {
    font-size: 18px;
}

.profile-disclaimer p {
    margin: 0;
    font-size: 10px;
}


/* ============================================================
   EMPTY / ERROR / ALERT
   ============================================================ */

.empty-state {
    background: var(--white);
    border: 1px dashed #cbd8e2;
    border-radius: var(--radius);
    padding: 45px 25px;
    text-align: center;
}

.empty-state.large {
    padding: 80px 25px;
}

.empty-icon {
    width: 70px;
    height: 70px;
    border-radius: 20px;
    margin: 0 auto 15px;
    display: grid;
    place-items: center;
    background: rgba(11, 132, 243, 0.08);
    font-size: 30px;
}

.empty-state h2,
.empty-state h3 {
    margin: 0 0 6px;
}

.empty-state p {
    color: var(--gray);
    font-size: 12px;
    margin: 0 0 20px;
}

.alert {
    display: flex;
    gap: 12px;
    padding: 17px;
    border-radius: 14px;
    margin-bottom: 20px;
}

.alert-error {
    background: rgba(228, 92, 92, 0.08);
    color: #8f3030;
    border: 1px solid rgba(228, 92, 92, 0.15);
}

.alert-icon {
    width: 34px;
    height: 34px;
    flex-shrink: 0;
    display: grid;
    place-items: center;
    border-radius: 10px;
    background: rgba(228, 92, 92, 0.12);
    font-weight: 800;
}

.alert strong {
    display: block;
    font-size: 12px;
}

.alert p {
    margin: 2px 0 0;
    font-size: 10px;
}


/* ============================================================
   ERROR PAGE
   ============================================================ */

.error-page {
    min-height: 70vh;
    display: grid;
    place-items: center;
    padding: 70px 0;
}

.error-card {
    width: min(600px, 100%);
    text-align: center;
    background: var(--white);
    border-radius: 25px;
    padding: 55px 30px;
    box-shadow: var(--shadow);
}

.error-number {
    color: rgba(11, 132, 243, 0.09);
    font-size: 100px;
    font-weight: 800;
    line-height: 1;
}

.error-icon {
    font-size: 45px;
    margin-top: -35px;
}

.error-card h1 {
    margin: 15px 0 8px;
}

.error-card p {
    color: var(--gray);
    font-size: 13px;
    margin: 0 auto 25px;
    max-width: 450px;
}


/* ============================================================
   FOOTER
   ============================================================ */

.site-footer {
    background: var(--dark);
    color: white;
    margin-top: 30px;
}

.footer-grid {
    display: grid;
    grid-template-columns: 1.5fr 1fr 1fr;
    gap: 50px;
    padding: 55px 0;
}

.footer-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 15px;
}

.site-footer p {
    color: rgba(255, 255, 255, 0.62);
    font-size: 11px;
    max-width: 360px;
    margin: 0;
}

.site-footer h3 {
    font-size: 13px;
    margin: 5px 0 14px;
}

.site-footer a {
    display: block;
    color: rgba(255, 255, 255, 0.62);
    font-size: 11px;
    margin-bottom: 7px;
}

.site-footer a:hover {
    color: white;
}

.footer-bottom {
    border-top: 1px solid rgba(255, 255, 255, 0.08);
    padding: 17px 0;
    color: rgba(255, 255, 255, 0.45);
    font-size: 9px;
}


/* ============================================================
   RESPONSIVE
   ============================================================ */

@media (max-width: 1050px) {

    .hero .container {
        grid-template-columns: 1fr;
        text-align: center;
    }

    .hero-content {
        display: flex;
        flex-direction: column;
        align-items: center;
    }

    .hero-visual {
        min-height: 330px;
    }

    .specialties-grid {
        grid-template-columns: repeat(2, 1fr);
    }

    .doctors-grid {
        grid-template-columns: repeat(2, 1fr);
    }

    .steps-grid {
        grid-template-columns: repeat(2, 1fr);
    }

    .filters-card {
        grid-template-columns: repeat(2, 1fr);
    }

    .search-field {
        grid-column: 1 / -1;
    }

    .filter-actions {
        grid-column: 1 / -1;
    }
}


@media (max-width: 760px) {

    .container {
        width: min(100% - 28px, 1180px);
    }

    .header-inner {
        min-height: 68px;
    }

    .main-nav {
        gap: 14px;
    }

    .main-nav a {
        font-size: 11px;
    }

    .brand-text small {
        display: none;
    }

    .brand-text strong {
        font-size: 13px;
    }

    .brand-icon {
        width: 40px;
        height: 40px;
        font-size: 21px;
    }

    .hero {
        padding: 55px 0 75px;
    }

    .hero h1 {
        font-size: 37px;
    }

    .hero p {
        font-size: 14px;
    }

    .hero-search {
        width: 100%;
        flex-direction: column;
        padding: 8px;
    }

    .search-input-wrapper {
        width: 100%;
    }

    .hero-search .primary-button {
        width: 100%;
    }

    .hero-visual {
        min-height: 280px;
    }

    .medical-circle {
        width: 250px;
    }

    .medical-cross {
        width: 95px;
        height: 95px;
        font-size: 70px;
        border-radius: 28px;
    }

    .floating-card {
        min-width: 170px;
        padding: 10px;
    }

    .floating-card-one {
        right: -4px;
    }

    .floating-card-two {
        left: -4px;
    }

    .stats-grid {
        grid-template-columns: 1fr;
    }

    .stats-section {
        margin-top: -25px;
    }

    .section {
        padding: 65px 0;
    }

    .section-heading {
        align-items: flex-start;
        flex-direction: column;
        gap: 15px;
    }

    .section-heading h2 {
        font-size: 25px;
    }

    .specialties-grid,
    .doctors-grid,
    .steps-grid {
        grid-template-columns: 1fr;
    }

    .page-hero-content {
        align-items: flex-start;
        flex-direction: column;
    }

    .page-hero h1 {
        font-size: 32px;
    }

    .filters-card {
        grid-template-columns: 1fr;
    }

    .search-field,
    .filter-actions {
        grid-column: auto;
    }

    .filter-actions {
        flex-direction: column;
    }

    .filter-actions .primary-button,
    .filter-actions .secondary-button {
        width: 100%;
    }

    .profile-layout {
        grid-template-columns: 1fr;
    }

    .profile-sidebar {
        position: static;
    }

    .profile-highlights {
        grid-template-columns: 1fr;
    }

    .details-grid {
        grid-template-columns: 1fr;
    }

    .footer-grid {
        grid-template-columns: 1fr;
        gap: 28px;
        padding: 40px 0;
    }
}


@media (max-width: 420px) {

    .main-nav {
        display: none;
    }

    .hero h1 {
        font-size: 32px;
    }

    .floating-card {
        min-width: 145px;
    }

    .floating-card strong {
        font-size: 10px;
    }

    .floating-card small {
        font-size: 8px;
    }

    .doctor-card-body {
        padding: 17px;
    }

    .profile-header {
        padding: 23px;
    }

    .profile-header h1 {
        font-size: 27px;
    }
}


/* ============================================================
   ACCESSIBILITY
   ============================================================ */

:focus-visible {
    outline: 3px solid rgba(11, 132, 243, 0.3);
    outline-offset: 3px;
}

::selection {
    background: rgba(11, 132, 243, 0.2);
}
"""


# ============================================================
# JAVASCRIPT
# ============================================================

APP_JS = r"""
document.addEventListener("DOMContentLoaded", function () {

    // Prevent double-submit on filter/search forms.
    const forms = document.querySelectorAll("form");

    forms.forEach(function (form) {
        form.addEventListener("submit", function () {

            const button = form.querySelector(
                'button[type="submit"]'
            );

            if (!button) {
                return;
            }

            button.dataset.originalText = button.textContent;

            button.textContent = "جارٍ البحث...";

            button.disabled = true;
        });
    });


    // Simple keyboard shortcut:
    // "/" focuses the main search field.
    document.addEventListener("keydown", function (event) {

        if (
            event.key === "/" &&
            document.activeElement.tagName !== "INPUT" &&
            document.activeElement.tagName !== "TEXTAREA" &&
            document.activeElement.tagName !== "SELECT"
        ) {

            const searchInput = document.querySelector(
                'input[name="q"]'
            );

            if (searchInput) {
                event.preventDefault();
                searchInput.focus();
            }
        }
    });

});
"""


# ============================================================
# JINJA ENVIRONMENT
# ============================================================

templates = Environment(
    loader=DictLoader(
        {
            "base.html": BASE_TEMPLATE,
            "home.html": HOME_TEMPLATE,
            "doctors.html": DOCTORS_TEMPLATE,
            "doctor_profile.html": PROFILE_TEMPLATE,
            "doctor_card.html": DOCTOR_CARD_TEMPLATE,
            "error.html": ERROR_TEMPLATE,
        }
    ),
    autoescape=select_autoescape(
        enabled_extensions=("html", "xml")
    ),
)


# ============================================================
# TEMPLATE RENDER HELPER
# ============================================================

def render_template(
    template_name: str,
    context: dict[str, Any],
) -> HTMLResponse:
    """
    Render a Jinja2 template.
    """

    from datetime import datetime

    final_context = {
        "app_name": APP_NAME,
        "current_year": datetime.now().year,
        **context,
    }

    template = templates.get_template(template_name)

    rendered = template.render(**final_context)

    return HTMLResponse(content=rendered)


# ============================================================
# ROUTES - STATIC
# ============================================================

@app.get(
    "/static/style.css",
    response_class=PlainTextResponse,
)
async def style_css() -> PlainTextResponse:
    """
    Serve CSS from Python string.
    """

    return PlainTextResponse(
        content=STYLE_CSS,
        media_type="text/css",
    )


@app.get(
    "/static/app.js",
    response_class=PlainTextResponse,
)
async def app_js() -> PlainTextResponse:
    """
    Serve JavaScript from Python string.
    """

    return PlainTextResponse(
        content=APP_JS,
        media_type="application/javascript",
    )


# ============================================================
# ROUTE - HOME
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse,
)
async def homepage(request: Request) -> HTMLResponse:
    """
    Homepage.
    """

    df = load_doctors()

    statistics = get_statistics(df)

    specialties = get_unique_values(
        df,
        "specialty",
    )

    # Display maximum 6 doctors on homepage.
    featured_df = df.head(6)

    featured_doctors = prepare_doctors(
        featured_df
    )

    return render_template(
        "home.html",
        {
            "request": request,
            "title": "الرئيسية",
            "statistics": statistics,
            "specialties": specialties,
            "featured_doctors": featured_doctors,
        },
    )


# ============================================================
# ROUTE - DOCTORS DIRECTORY
# ============================================================

@app.get(
    "/doctors",
    response_class=HTMLResponse,
)
async def doctors_page(
    request: Request,
    q: str = Query(
        default="",
        max_length=100,
    ),
    specialty: str = Query(
        default="",
        max_length=100,
    ),
    area: str = Query(
        default="",
        max_length=100,
    ),
) -> HTMLResponse:
    """
    Doctors directory.

    Example:
    /doctors?q=أحمد
    /doctors?specialty=باطنة
    /doctors?q=أحمد&specialty=باطنة&area=دمياط
    """

    df = load_doctors()

    specialties = get_unique_values(
        df,
        "specialty",
    )

    areas = get_unique_values(
        df,
        "area",
    )

    filtered_df = filter_doctors(
        df=df,
        query=q,
        specialty=specialty,
        area=area,
    )

    doctors = prepare_doctors(
        filtered_df
    )

    return render_template(
        "doctors.html",
        {
            "request": request,
            "title": "الأطباء",
            "doctors": doctors,
            "result_count": len(doctors),
            "specialties": specialties,
            "areas": areas,
            "q": safe_text(q),
            "specialty": safe_text(specialty),
            "area": safe_text(area),
            "csv_error": get_csv_error(),
        },
    )


# ============================================================
# ROUTE - DOCTOR PROFILE
# ============================================================

@app.get(
    "/doctors/{doctor_id}",
    response_class=HTMLResponse,
)
async def doctor_profile(
    request: Request,
    doctor_id: str,
) -> HTMLResponse:
    """
    Individual doctor profile.

    Uses the ID from doctors.csv.
    """

    df = load_doctors()

    doctor = find_doctor_by_id(
        df,
        doctor_id,
    )

    if doctor is None:

        return render_template(
            "error.html",
            {
                "request": request,
                "title": "الطبيب غير موجود",
                "error_code": "404",
                "icon": "🩺",
                "message": (
                    "عذرًا، الطبيب غير موجود في دليل الأطباء."
                ),
                "button_url": "/doctors",
                "button_text": "العودة إلى الأطباء",
            },
        )

    return render_template(
        "doctor_profile.html",
        {
            "request": request,
            "title": doctor.get(
                "name",
                "ملف الطبيب",
            ),
            "doctor": doctor,
        },
    )


# ============================================================
# ROUTE - MANUAL CSV RELOAD
# ============================================================

@app.get("/reload")
async def reload_csv() -> RedirectResponse:
    """
    Force reload doctors.csv.

    After reload, return to doctors directory.
    """

    load_doctors(force_reload=True)

    return RedirectResponse(
        url="/doctors",
        status_code=303,
    )


# ============================================================
# CUSTOM 404
# ============================================================

@app.exception_handler(404)
async def not_found_handler(
    request: Request,
    exc: HTTPException,
) -> HTMLResponse:
    """
    Arabic 404 page.
    """

    return render_template(
        "error.html",
        {
            "request": request,
            "title": "الصفحة غير موجودة",
            "error_code": "404",
            "icon": "🔎",
            "message": (
                "عذرًا، الصفحة التي تبحث عنها غير موجودة "
                "أو ربما تم تغيير رابطها."
            ),
            "button_url": "/",
            "button_text": "العودة إلى الرئيسية",
        },
    )


# ============================================================
# CUSTOM 500
# ============================================================

@app.exception_handler(Exception)
async def general_exception_handler(
    request: Request,
    exc: Exception,
) -> HTMLResponse:
    """
    Generic error page.

    Does not expose traceback to users.
    """

    return render_template(
        "error.html",
        {
            "request": request,
            "title": "حدث خطأ غير متوقع",
            "error_code": "500",
            "icon": "⚠️",
            "message": (
                "حدث خطأ أثناء معالجة الطلب. "
                "يرجى المحاولة مرة أخرى."
            ),
            "button_url": "/",
            "button_text": "العودة إلى الرئيسية",
        },
    )


# ============================================================
# STARTUP
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("دليل أطباء دمياط")
    print("=" * 60)
    print()
    print(f"Local URL: http://{HOST}:{PORT}")
    print(f"CSV File : {CSV_PATH}")
    print()
    print("اضغط CTRL+C لإيقاف السيرفر.")
    print()

    # Initial CSV load.
    load_doctors()

    if get_csv_error():
        print("CSV WARNING:")
        print(get_csv_error())
        print()

    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info",
    )