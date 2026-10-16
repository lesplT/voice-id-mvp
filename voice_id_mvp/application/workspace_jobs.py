"""Bounded in-memory task statuses; persistent recordings live in JournalStore."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Lock
from time import time
from uuid import uuid4


class QueueFull(RuntimeError):
    """Purpose:
        Represent rejection when the bounded recognition queue has no free slot.

    Ingress:
        A queue-capacity explanation.

    Egress:
        An exception handled as a retryable HTTP error.
    """
    pass


class JobManager:
    """Purpose:
        Run bounded recognition jobs while keeping task status separate from persistent history.

    Ingress:
        Capacity, retained-result limit, and in-memory status lifetime.

    Egress:
        Thread-safe queued/running/completed job snapshots and a single-worker executor.
    """
    def __init__(self, capacity=4, history=20, ttl=3600):
        """Purpose:
            Initialize queue limits, synchronization, and the background worker.

        Ingress:
            Maximum unfinished jobs, retained statuses, and status TTL in seconds.

        Egress:
            A ready JobManager with no jobs submitted.
        """
        self.capacity, self.history, self.ttl = capacity, history, ttl
        self._jobs = {}
        self._lock = Lock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="voice-workspace")

    def _prune(self):
        """Purpose:
            Evict expired or excess completed job snapshots, never persistent recordings.

        Ingress:
            Current in-memory jobs and configured retention limits; caller holds the lock.

        Egress:
            None; eligible completed/failed entries are removed from memory.
        """
        completed = sorted((j for j in self._jobs.values() if j['status'] in {'done','failed'}), key=lambda j:j['created_at'])
        for j in completed:
            if time()-j['created_at'] > self.ttl or len(self._jobs) >= self.history:
                self._jobs.pop(j['id'],None)

    def submit(self, operation, callback):
        """Purpose:
            Queue one callback if the unfinished-job capacity permits it.

        Ingress:
            An operation label and zero-argument processing callback.

        Egress:
            A copied initial job snapshot; raises QueueFull when capacity is exhausted.
        """
        with self._lock:
            self._prune()
            if sum(j['status'] in {'queued','running'} for j in self._jobs.values()) >= self.capacity:
                raise QueueFull("Очередь заполнена. Дождитесь завершения текущей записи.")
            job = {'id':uuid4().hex, 'operation':operation, 'status':'queued', 'created_at':time(), 'result':None, 'error':None}
            self._jobs[job['id']] = job
        self._pool.submit(self._run,job['id'],callback)
        return self.get(job['id'])

    def _run(self, job_id, callback):
        """Purpose:
            Execute a queued callback and record completion or failure.

        Ingress:
            A job identifier and its processing callback.

        Egress:
            None; status, result/error, and finish time are updated under the lock.
        """
        with self._lock:
            self._jobs[job_id]['status'] = 'running'
        try:
            result = callback()
            with self._lock:
                self._jobs[job_id].update(status='done',result=result)
        except Exception as exc:
            with self._lock:
                self._jobs[job_id].update(status='failed',error=str(exc))
        finally:
            with self._lock:
                self._jobs[job_id]['finished_at'] = time()

    def get(self, job_id):
        """Purpose:
            Return an isolated snapshot of a retained job.

        Ingress:
            A job identifier.

        Egress:
            A deep-copied status dictionary; raises KeyError after eviction or for unknown IDs.
        """
        with self._lock:
            self._prune()
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return deepcopy(self._jobs[job_id])

    def shutdown(self):
        """Purpose:
            Wait for submitted work and release the executor.

        Ingress:
            The manager's existing background executor.

        Egress:
            None; pending work is allowed to finish.
        """
        self._pool.shutdown(wait=True)

    def forget_journal(self, recording_id):
        """Purpose:
            Forget completed in-memory results associated with an explicitly deleted journal entry.

        Ingress:
            A journal recording identifier.

        Egress:
            None; matching completed job snapshots are removed.
        """
        with self._lock:
            for job_id, job in list(self._jobs.items()):
                if (job.get('result') or {}).get('journal_id') == recording_id and job['status'] in {'done','failed'}:
                    self._jobs.pop(job_id)
