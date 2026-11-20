"""Builds the list of Source objects from sources.yaml."""

from pathlib import Path

import yaml

from app.ingest.base import Source
from app.ingest.sources import (
    ArbeitnowSource,
    AshbySource,
    GreenhouseSource,
    LeverSource,
    RemotiveSource,
)


def load_sources(path: str | Path) -> list[Source]:
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")

    unknown = set(config) - {"remotive", "arbeitnow", "greenhouse", "lever", "ashby"}
    if unknown:
        raise ValueError(f"{path}: unknown source types {sorted(unknown)}")

    sources: list[Source] = []
    if (config.get("remotive") or {}).get("enabled"):
        sources.append(RemotiveSource())
    arbeitnow = config.get("arbeitnow") or {}
    if arbeitnow.get("enabled"):
        sources.append(ArbeitnowSource(max_pages=int(arbeitnow.get("max_pages", 3))))
    for board in config.get("greenhouse") or []:
        sources.append(GreenhouseSource(board=str(board)))
    for entry in config.get("lever") or []:
        if isinstance(entry, str):
            sources.append(LeverSource(company=entry))
        else:
            sources.append(LeverSource(company=entry["slug"], display_name=entry.get("name")))

    for entry in config.get("ashby") or []:
        if isinstance(entry, str):
            sources.append(AshbySource(org=entry))
        else:
            sources.append(AshbySource(org=entry["slug"], display_name=entry.get("name")))

    keys = [s.key for s in sources]
    duplicates = {k for k in keys if keys.count(k) > 1}
    if duplicates:
        raise ValueError(f"{path}: duplicate sources {sorted(duplicates)}")
    return sources
