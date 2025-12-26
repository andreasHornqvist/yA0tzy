import asyncio
import time

from yatzy_az.server.batcher import BatcherStats, ModelStats
from yatzy_az.server.metrics import MetricsSnapshot, format_prometheus
from yatzy_az.server.metrics_server import start_metrics_server


def test_format_prometheus_contains_expected_names() -> None:
    bs = BatcherStats()
    bs.requests_total = 10
    bs.batches_total = 3
    bs.by_model[0] = ModelStats(requests_total=6, batches_total=2, batch_hist={2: 1, 4: 1})
    bs.by_model[1] = ModelStats(requests_total=4, batches_total=1, batch_hist={4: 1})
    snap = MetricsSnapshot(now_s=100.0, start_s=50.0, queue_depth=7, batcher=bs)
    txt = format_prometheus(snap)

    assert "yatzy_infer_uptime_seconds" in txt
    assert "yatzy_infer_queue_depth" in txt
    assert "yatzy_infer_requests_total" in txt
    assert "yatzy_infer_batches_total" in txt
    assert 'yatzy_infer_batch_size_bucket{model_id="0",le="2"}' in txt
    assert 'yatzy_infer_batch_size_bucket{model_id="1",le="+Inf"}' in txt


def test_metrics_server_scrape_smoke() -> None:
    asyncio.run(_metrics_server_scrape_smoke())


async def _metrics_server_scrape_smoke() -> None:
    bs = BatcherStats()
    bs.requests_total = 1
    bs.batches_total = 1
    bs.by_model[0] = ModelStats(requests_total=1, batches_total=1, batch_hist={1: 1})

    start = time.time()

    def snapshot() -> MetricsSnapshot:
        return MetricsSnapshot(now_s=time.time(), start_s=start, queue_depth=0, batcher=bs)

    # Bind to ephemeral port.
    srv = await start_metrics_server("127.0.0.1:0", snapshot)
    host, port = srv.sockets[0].getsockname()[:2]

    try:
        r, w = await asyncio.open_connection(host, port)
        w.write(b"GET /metrics HTTP/1.1\r\nHost: localhost\r\n\r\n")
        await w.drain()
        data = await r.read(65536)
        assert b"HTTP/1.1 200" in data
        assert b"yatzy_infer_requests_total" in data
    finally:
        srv.close()
        await srv.wait_closed()
