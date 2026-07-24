from pathlib import Path

from tools import release_gate_scan


class _Response:
    status_code = 200
    headers: dict[str, str] = {}
    links = {"next": {"url": "https://example.test/api/models?cursor=next"}}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> list[dict]:
        return [{"id": "owner/first"}]


class _FileResponse:
    encoding = "utf-8"

    def __init__(self, status_code: int, body: bytes = b"", **headers: str) -> None:
        self.status_code = status_code
        self.headers = headers
        self._body = body

    def iter_content(self, chunk_size: int):
        yield self._body

    def close(self) -> None:
        return None


def test_inventory_page_follows_link_cursor(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    def get(url: str, **kwargs) -> _Response:
        calls.append((url, kwargs["params"]))
        return _Response()

    monkeypatch.setattr(release_gate_scan.requests, "get", get)

    page, next_url = release_gate_scan.fetch_inventory_page("token", limit=3)
    release_gate_scan.fetch_inventory_page("token", next_url, limit=3)

    assert page == [{"id": "owner/first"}]
    assert next_url == "https://example.test/api/models?cursor=next"
    assert calls[0][0] == release_gate_scan.HF_API
    assert calls[0][1] is not None
    assert calls[1] == (next_url, None)


def test_inventory_uses_distinct_cursor_pages(tmp_path: Path, monkeypatch) -> None:
    pages = {
        None: ([{"id": "owner/first", "siblings": []}], "cursor-2"),
        "cursor-2": ([{"id": "owner/second", "siblings": []}], None),
    }

    def fetch(token: str, url: str | None = None, limit: int = 1000):
        return pages[url]

    monkeypatch.setattr(release_gate_scan, "fetch_inventory_page", fetch)
    destination = tmp_path / "inventory.json"

    inventory = release_gate_scan.build_inventory(destination, "token", lambda _: None)

    assert [item["repo"] for item in inventory] == ["owner/first", "owner/second"]
    assert destination.exists()


def test_repo_fetch_waits_out_rate_limit(monkeypatch) -> None:
    responses = [
        _FileResponse(429, RateLimit='"resolvers";r=0;t=2'),
        _FileResponse(200, b'{"ok": true}'),
    ]
    sleeps: list[float] = []

    monkeypatch.setattr(release_gate_scan.RESOLVER_LIMITER, "wait", lambda: None)
    monkeypatch.setattr(release_gate_scan.time, "sleep", sleeps.append)
    monkeypatch.setattr(
        release_gate_scan.requests,
        "get",
        lambda *args, **kwargs: responses.pop(0),
    )

    content = release_gate_scan.fetch_gate_text("owner/model", "config.json", 1024)

    assert content == '{"ok": true}'
    assert sleeps == [3.0]
