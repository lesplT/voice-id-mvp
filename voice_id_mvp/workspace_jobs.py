"""Bounded in-memory task statuses; persistent recordings live in JournalStore."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Lock
from time import time
from uuid import uuid4


class QueueFull(RuntimeError):
    pass


class JobManager:
    def __init__(self, capacity=4, history=20, ttl=3600):
        self.capacity, self.history, self.ttl = capacity, history, ttl
        self._jobs = {}
        self._lock = Lock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="voice-workspace")

    def _prune(self):
        completed = sorted((j for j in self._jobs.values() if j['status'] in {'done','failed'}), key=lambda j:j['created_at'])
        for j in completed:
            if time()-j['created_at'] > self.ttl or len(self._jobs) >= self.history:
                self._jobs.pop(j['id'],None)

    def submit(self, operation, callback):
        with self._lock:
            self._prune()
            if sum(j['status'] in {'queued','running'} for j in self._jobs.values()) >= self.capacity:
                raise QueueFull("Очередь заполнена. Дождитесь завершения текущей записи.")
            job = {'id':uuid4().hex, 'operation':operation, 'status':'queued', 'created_at':time(), 'result':None, 'error':None}
            self._jobs[job['id']] = job
        self._pool.submit(self._run,job['id'],callback)
        return self.get(job['id'])

    def _run(self, job_id, callback):
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
        with self._lock:
            self._prune()
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return deepcopy(self._jobs[job_id])

    def shutdown(self):
        self._pool.shutdown(wait=True)

    def forget_journal(self, recording_id):
        with self._lock:
            for job_id, job in list(self._jobs.items()):
                if (job.get('result') or {}).get('journal_id') == recording_id and job['status'] in {'done','failed'}:
                    self._jobs.pop(job_id)
