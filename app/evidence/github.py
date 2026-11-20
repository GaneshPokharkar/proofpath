"""Read a student's public GitHub repos through the REST API.

Unauthenticated calls are limited to 60 an hour, enough for one student with a few repos;
set JOBPILOT_GITHUB_TOKEN (no scopes needed for public data) for 5,000 an hour.
"""

import base64
from dataclasses import dataclass, field
from datetime import datetime

import httpx

API = "https://api.github.com"

# Files whose contents declare dependencies or prove a tool is in use.
MANIFESTS = (
    "requirements.txt",
    "pyproject.toml",
    "package.json",
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    "compose.yaml",
)


class GitHubError(RuntimeError):
    pass


@dataclass
class RepoSnapshot:
    full_name: str
    url: str
    description: str | None
    is_fork: bool
    stars: int
    pushed_at: datetime | None
    languages: dict[str, int] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)  # manifest path -> text
    workflows: list[str] = field(default_factory=list)  # .github/workflows file names
    readme: str = ""


class GitHubClient:
    def __init__(self, client: httpx.Client, token: str | None = None):
        self._client = client
        self._headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        if token:
            self._headers["Authorization"] = f"Bearer {token}"

    def _get(self, path: str, params: dict | None = None, allow_404: bool = False):
        r = self._client.get(f"{API}{path}", params=params, headers=self._headers)
        if r.status_code == 404 and allow_404:
            return None
        if r.status_code == 403 and r.headers.get("x-ratelimit-remaining") == "0":
            raise GitHubError("GitHub rate limit reached; set JOBPILOT_GITHUB_TOKEN or wait an hour")
        if r.status_code != 200:
            raise GitHubError(f"GitHub {r.status_code} for {path}: {r.text[:200]}")
        return r.json()

    def _file(self, full_name: str, path: str) -> str | None:
        data = self._get(f"/repos/{full_name}/contents/{path}", allow_404=True)
        if not data or data.get("encoding") != "base64":
            return None
        return base64.b64decode(data["content"]).decode("utf-8", errors="replace")

    def list_repos(self, login: str, limit: int = 15) -> list[dict]:
        """The user's own repos, most recently pushed first."""
        repos = self._get(f"/users/{login}/repos", {"type": "owner", "sort": "pushed", "per_page": 100})
        return repos[:limit]

    def snapshot(self, repo: dict) -> RepoSnapshot:
        full_name = repo["full_name"]
        pushed = repo.get("pushed_at")
        snap = RepoSnapshot(
            full_name=full_name,
            url=repo["html_url"],
            description=repo.get("description"),
            is_fork=bool(repo.get("fork")),
            stars=int(repo.get("stargazers_count") or 0),
            pushed_at=datetime.fromisoformat(pushed.replace("Z", "+00:00")) if pushed else None,
        )
        snap.languages = self._get(f"/repos/{full_name}/languages") or {}
        readme = self._get(f"/repos/{full_name}/readme", allow_404=True)
        if readme and readme.get("encoding") == "base64":
            snap.readme = base64.b64decode(readme["content"]).decode("utf-8", errors="replace")
        # One listing of the root tells us which manifests exist, so absent ones cost no calls.
        root = self._get(f"/repos/{full_name}/contents/", allow_404=True) or []
        present = {item["name"] for item in root if item.get("type") == "file"}
        for path in MANIFESTS:
            if path in present:
                text = self._file(full_name, path)
                if text is not None:
                    snap.files[path] = text
        if any(item.get("name") == ".github" for item in root):
            workflows = self._get(f"/repos/{full_name}/contents/.github/workflows", allow_404=True) or []
            snap.workflows = [w["name"] for w in workflows if w.get("type") == "file"]
        return snap
