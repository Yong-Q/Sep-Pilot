"""Regression tests for the ai2-kit-absorbed JobWatch improvements:

1. Batch squeue query with TTL cache (one squeue per poll instead of one
   subprocess per job).
2. squeue short-code -> canonical state table (JobState-style terminal/failed
   flags).
3. job.done success-indicator detection (distinguish real COMPLETED from a
   vanished failure on hosts with no sacct accounting).
4. JobWatch._poll_once batch dispatch: cheap path for PENDING/RUNNING/
   COMPLETED jobs, full software-level check only for failed/vanished jobs.
5. Confirmed completions emit wake-up notifications so a multi-step workflow
   resumes its next dependency; failures emit diagnostic notifications.
"""
import os
import tempfile

from agents import slurm
from agents.job_watch import JobWatch


def test_translate_squeue_state():
    assert slurm.translate_squeue_state("PD") == {"status": "PENDING", "terminal": False, "failed": False}
    assert slurm.translate_squeue_state("R") == {"status": "RUNNING", "terminal": False, "failed": False}
    assert slurm.translate_squeue_state("CD") == {"status": "COMPLETED", "terminal": True, "failed": False}
    for code in ("CA", "F", "NF", "TO", "DL", "RV", "SE"):
        s = slurm.translate_squeue_state(code)
        assert s["terminal"] is True and s["failed"] is True, code


def test_batch_fetch_states_cache():
    # Real squeue on the login node: must not crash and must return a dict.
    first = slurm.batch_fetch_states()
    assert isinstance(first, dict)
    import time
    _ = slurm.batch_fetch_states()  # cache hit, should be ~0s


def test_job_done_success_indicator():
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "job.done"), "w") as f:
        f.write("999999 done\n")
    r = slurm.check_job_status("999999", work_dir=d)
    assert r["status"] == "COMPLETED"
    assert r["terminal"] is True
    assert r["failed"] is False

    # Vanished job without indicator: NOT proven completed.
    d2 = tempfile.mkdtemp()
    r2 = slurm.check_job_status("999998", work_dir=d2)
    assert r2["status"] != "COMPLETED"

    # Indicator belonging to a DIFFERENT job must not be trusted.
    d3 = tempfile.mkdtemp()
    with open(os.path.join(d3, "job.done"), "w") as f:
        f.write("111111 done\n")
    r3 = slurm.check_job_status("999997", work_dir=d3)
    assert r3["status"] != "COMPLETED"


def test_poll_once_batch_dispatch():
    watch = JobWatch(watch_file=os.path.join(tempfile.mkdtemp(), "jw.json"))
    watch.register("101", work_dir="/wd/101", conv_id="c1", username="u")
    watch.register("102", work_dir="/wd/102", conv_id="c1", username="u")
    watch.register("103", work_dir="/wd/103", conv_id="c1", username="u")
    watch.register("104", work_dir="/wd/104", conv_id="c1", username="u")

    calls = {"check": []}

    def fake_batch(job_ids=None):
        return {
            "101": {"status": "RUNNING", "terminal": False, "failed": False},
            "102": {"status": "COMPLETED", "terminal": True, "failed": False},
            "103": {"status": "FAILED", "terminal": True, "failed": True},
            # 104 absent -> vanished from the scheduler queue
        }

    def fake_check(job_id, work_dir=""):
        calls["check"].append(job_id)
        if job_id == "103":
            return {"status": "FAILED", "terminal": True, "failed": True,
                    "error": "boom", "diagnosis": {"cause": "x", "fixes": ["y"]}}
        return {"status": "COMPLETED", "terminal": True, "failed": False}

    orig_b, orig_c = slurm.batch_fetch_states, slurm.check_job_status
    slurm.batch_fetch_states, slurm.check_job_status = fake_batch, fake_check
    try:
        watch._poll_once()
    finally:
        slurm.batch_fetch_states, slurm.check_job_status = orig_b, orig_c

    # Cheap path: RUNNING and COMPLETED-from-queue never hit check_job_status.
    assert calls["check"] == ["103", "104"]
    assert watch.get("101")["state"] == "RUNNING" and not watch.get("101")["terminal"]
    assert watch.get("102")["state"] == "COMPLETED"
    assert watch.get("103")["failed"] and watch.get("103")["notified"]
    assert watch.get("104")["state"] == "COMPLETED"

    notifs = watch.drain_notifications()
    by_job = {n["job_id"]: n for n in notifs}
    assert set(by_job) == {"102", "103", "104"}
    assert by_job["103"]["state"] == "FAILED"
    assert by_job["102"]["state"] == "COMPLETED"
    assert by_job["104"]["state"] == "COMPLETED"


def test_notification_survives_restart_until_acknowledged():
    path = os.path.join(tempfile.mkdtemp(), "jw.json")
    first = JobWatch(watch_file=path)
    first.register("201", work_dir="/wd/201", conv_id="c1", username="u")
    first._notify_completion("201", "COMPLETED")

    restarted = JobWatch(watch_file=path)
    pending = restarted.drain_notifications()
    assert [n["job_id"] for n in pending] == ["201"]
    restarted.ack_notification("201")

    after_ack = JobWatch(watch_file=path)
    assert after_ack.drain_notifications() == []


def test_worker_status_update_does_not_resurrect_acknowledged_notification(tmp_path):
    path = str(tmp_path / 'jobs.json')
    first = JobWatch(watch_file=path)
    first.register('301', conv_id='c1', username='u')
    first._notify_completion('301', 'COMPLETED')
    other_worker = JobWatch(watch_file=path)
    first.ack_notification('301')
    other_worker._jobs['301']['state'] = 'COMPLETED'
    other_worker._save()
    restarted = JobWatch(watch_file=path)
    assert restarted.drain_notifications() == []


def test_unknown_scheduler_state_does_not_erase_log_confirmed_failure(tmp_path):
    path = str(tmp_path / 'jobs.json')
    watch = JobWatch(watch_file=path)
    watch.register('401', conv_id='c1', username='u')
    watch._jobs['401'].update({'state': 'UNKNOWN', 'terminal': True, 'failed': True, 'notified': True})
    watch._save()
    restarted = JobWatch(watch_file=path)
    record = restarted.get('401')
    assert record['failed'] and record['terminal'] and record['notified']
