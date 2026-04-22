from fastapi import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

STEPS_TOTAL = Counter("yogacara_steps_total", "Total decision steps")
REWARD_TOTAL = Gauge("yogacara_reward_total", "Cumulative reward")
MANAS_INTERCEPTIONS = Counter("yogacara_manas_interceptions_total", "Manas interception count")
MEMORY_HITS = Counter("yogacara_memory_hits_total", "Seed retrieval hits")
ALIGNMENT_SCORE = Gauge("yogacara_alignment_score", "Current alignment score")
STEP_DURATION = Histogram("yogacara_step_duration_seconds", "Step execution latency")


def expose_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def record_step(reward: float, alignment: float, duration: float):
    STEPS_TOTAL.inc()
    REWARD_TOTAL.set(reward)
    ALIGNMENT_SCORE.set(alignment)
    STEP_DURATION.observe(duration)


def record_interception():
    MANAS_INTERCEPTIONS.inc()


def record_memory_hit(count: int):
    MEMORY_HITS.inc(count)
