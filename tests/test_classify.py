"""Field and level classification from job titles, including the mistakes found on live data."""

import pytest

from app.ingest.classify import classify_field, classify_level

FIELDS = [
    ("Senior Software Engineer - Observability", "software"),
    ("SDE II - Data Engineer", "data_ai"),
    ("Data Analyst Intern", "data_ai"),
    ("Business Analyst", "data_ai"),
    ("Workplace Operations Analyst", "other"),
    ("Senior Financial Analyst", "other"),
    ("Front End Engineer, Marketing", "frontend"),
    ("Account Executive", "other"),
    ("Senior Software Engineer, Cloud Payments", "software"),
    ("Cloud Engineer", "devops_cloud"),
    ("Site Reliability Engineer", "devops_cloud"),
    ("Vulnerability Management Engineer", "security"),
    ("Android Developer", "mobile"),
    ("QA Automation Engineer", "qa"),
    ("Product Designer", "design"),
    ("Senior Product Manager", "product"),
]
LEVELS = [
    ("Data Analyst Intern", "intern"),
    ("Summer Internship 2027 - Software", "intern"),
    ("Associate Software Engineer", "entry"),
    ("Junior Backend Developer", "entry"),
    ("SDE 1, Backend", "entry"),
    ("Associate Director, Strategy", "senior"),
    ("Senior Associate - Workspace", "senior"),
    ("SDE II - Data Engineer", "senior"),
    ("Software Engineer", "mid"),
]


@pytest.mark.parametrize(("title", "field"), FIELDS)
def test_field(title, field):
    assert classify_field(title) == field


@pytest.mark.parametrize(("title", "level"), LEVELS)
def test_level(title, level):
    assert classify_level(title) == level
