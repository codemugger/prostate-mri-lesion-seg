#!/usr/bin/env python3
"""Operator tool for the prostate MRI CARPL stack.

Run inside the sender container, which has Redis access and the shared
helpers::

    docker exec prostate-dicom-sender python /opt/carpl/carpl_ctl.py status
    docker exec prostate-dicom-sender python /opt/carpl/carpl_ctl.py jobs --limit 20
    docker exec prostate-dicom-sender python /opt/carpl/carpl_ctl.py requeue --queue send --job-id <id>
    docker exec prostate-dicom-sender python /opt/carpl/carpl_ctl.py requeue --queue jobs --all

Output shows only technical identifiers (job ids, UIDs, states); no patient
demographics.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.environ.get("CARPL_LIB_DIR", "/opt/carpl/lib"))

from carpl_common import (  # noqa: E402
    JOBS,
    JOBS_FAILED,
    JOBS_PROCESSING,
    SEND,
    SEND_FAILED,
    SEND_PROCESSING,
    JobPaths,
    connect_redis,
    data_paths_from_env,
    heartbeat_age_seconds,
    read_json,
    study_latest_key,
    update_job_state,
)

QUEUES = (JOBS, JOBS_PROCESSING, JOBS_FAILED, SEND, SEND_PROCESSING, SEND_FAILED)
SERVICES = ("dicom-listener", "ai-worker", "dicom-sender")


def cmd_status(broker, paths) -> int:
    print("Queues:")
    for queue in QUEUES:
        print(f"  {queue:28s} {broker.length(queue)}")
    print("Heartbeats (seconds since last beat):")
    for service in SERVICES:
        age = heartbeat_age_seconds(paths, service)
        print(f"  {service:28s} {'never' if age is None else f'{age:.0f}'}")
    counts: "dict[str, int]" = {}
    if paths.jobs.exists():
        for job_dir in paths.jobs.iterdir():
            if job_dir.is_dir():
                state = (read_json(JobPaths(job_dir).state_json, default={}) or {}).get("state", "unknown")
                counts[state] = counts.get(state, 0) + 1
    print("Job folders by state:")
    for state, count in sorted(counts.items()):
        print(f"  {state:28s} {count}")
    return 0


def cmd_jobs(paths, limit: int) -> int:
    if not paths.jobs.exists():
        print("No jobs.")
        return 0
    job_dirs = sorted((p for p in paths.jobs.iterdir() if p.is_dir()), reverse=True)[:limit]
    print(f"{'job_id':28s} {'state':13s} {'status':22s} {'instances':>9s}  study_uid")
    for job_dir in job_dirs:
        job = JobPaths(job_dir)
        state = read_json(job.state_json, default={}) or {}
        manifest = read_json(job.job_json, default={}) or {}
        print(
            f"{job_dir.name:28s} {state.get('state', '?'):13s} {state.get('status', ''):22s} "
            f"{manifest.get('instance_count', ''):>9}  {manifest.get('study_instance_uid', '')}"
        )
    return 0


def cmd_requeue(broker, paths, queue: str, job_id: "str | None", requeue_all: bool) -> int:
    if not job_id and not requeue_all:
        print("Specify --job-id or --all", file=sys.stderr)
        return 2
    failed, target = (JOBS_FAILED, JOBS) if queue == "jobs" else (SEND_FAILED, SEND)
    client = broker._client  # raw list access for an operator tool
    moved = 0
    for raw in client.lrange(failed, 0, -1):
        payload = json.loads(raw)
        if job_id and payload.get("job_id") != job_id:
            continue
        payload.pop("reason", None)
        job = paths.job(payload["job_id"])
        if job.root.is_dir():
            if queue == "jobs":
                update_job_state(job, "queued", attempts=0, requeued=True)
                if payload.get("study_uid"):
                    broker.set(study_latest_key(payload["study_uid"]), payload["job_id"], 86400)
            else:
                update_job_state(job, "send_queued", requeued=True)
        client.lrem(failed, 1, raw)
        broker.push(target, payload)
        moved += 1
        print(f"Re-queued {payload['job_id']} onto {target}")
    if not moved:
        print("Nothing matched.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    jobs = sub.add_parser("jobs")
    jobs.add_argument("--limit", type=int, default=20)
    requeue = sub.add_parser("requeue")
    requeue.add_argument("--queue", choices=("jobs", "send"), required=True)
    requeue.add_argument("--job-id")
    requeue.add_argument("--all", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    paths = data_paths_from_env()
    if args.command == "jobs":
        return cmd_jobs(paths, args.limit)
    broker = connect_redis(logging.getLogger("carpl_ctl"), wait_seconds=10)
    if args.command == "status":
        return cmd_status(broker, paths)
    return cmd_requeue(broker, paths, args.queue, args.job_id, args.all)


if __name__ == "__main__":
    raise SystemExit(main())
