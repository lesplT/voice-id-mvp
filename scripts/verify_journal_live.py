"""Verify live persistence/UNKNOWN; optionally save the user's test4 into the journal.

Deletes only the synthetic recording this invocation creates. Never deletes user entries.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from voice_id_mvp.audio import write_wav_slice

parser = argparse.ArgumentParser()
parser.add_argument('--base-url', default='http://127.0.0.1:8000')
parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/journal-check')
parser.add_argument('--real-dialogue', action='store_true', help='Save test4sound.mp3 in the live journal')
parser.add_argument('--check-saved', action='store_true', help='Verify the previously saved record after restarting workspace')
args = parser.parse_args()
output = args.output_dir
output.mkdir(parents=True, exist_ok=True)
report_path = output / 'live-report.json'


def submit(client, name, data):
    response = client.post('/api/jobs', data={'operation': 'dialogue', 'use_dictionary': 'false'},
                           files={'audio': (name, data, 'application/octet-stream')})
    response.raise_for_status()
    job = response.json()
    started = time.monotonic()
    while job['status'] not in {'done', 'failed'}:
        if time.monotonic() - started > 600:
            raise RuntimeError('Journal verification exceeded 600 seconds')
        time.sleep(.4)
        response = client.get('/api/jobs/' + job['id'])
        response.raise_for_status()
        job = response.json()
    if job['status'] != 'done':
        raise RuntimeError(job['error'])
    return job, time.monotonic() - started


with httpx.Client(base_url=args.base_url, timeout=30, trust_env=False) as client:
    if args.check_saved:
        report = json.loads(report_path.read_text(encoding='utf-8'))
        entry = client.get('/api/journal/' + report['recording_id'])
        entry.raise_for_status()
        assert entry.json()['status'] == 'done'
        audio = client.get('/api/journal/' + report['recording_id'] + '/audio')
        audio.raise_for_status()
        assert hashlib.sha256(audio.content).hexdigest() == report['audio_sha256']
        assert client.get('/api/jobs/' + report['job_id']).status_code == 404
        report['survives_actual_workspace_restart'] = True
    else:
        report = {}
        with tempfile.TemporaryDirectory(prefix='voice-journal-check-') as temporary:
            clip = Path(temporary) / 'unknown.wav'
            write_wav_slice(ROOT / 'data/synthetic_russian_tts.wav', clip, 1, 4)
            job, _ = submit(client, 'journal-verification-unknown.wav', clip.read_bytes())
            synthetic_id = job['result']['journal_id']
            try:
                segments = job['result']['segments']
                assert any(s['speaker_id'] == 'UNKNOWN' and s['text'].strip() for s in segments)
                saved = client.get('/api/journal/' + synthetic_id).json()
                assert saved['result']['segments'] == segments
                assert client.get('/api/journal/' + synthetic_id + '/audio').content == clip.read_bytes()
                report['unregistered_voice_text_and_audio_saved'] = True
            finally:
                client.delete('/api/journal/' + synthetic_id).raise_for_status()
            assert client.get('/api/journal/' + synthetic_id).status_code == 404
            assert client.get('/api/journal/' + synthetic_id + '/audio').status_code == 404
            assert client.get('/api/jobs/' + job['id']).status_code == 404
            report['synthetic_recording_deleted'] = True
        if args.real_dialogue:
            source = Path('C:/Users/PC/Downloads/test4sound.mp3')
            job, seconds = submit(client, source.name, source.read_bytes())
            result = job['result']
            assert any(s['speaker_id'] == 'platon' and 'Круассан вам подогреть?' in s['text'] for s in result['segments'])
            assert any(s['speaker_id'] == 'prokhor' and 'Да, если можно' in s['text'] for s in result['segments'])
            report.update(recording_id=result['journal_id'], job_id=job['id'], real_dialogue_seconds=round(seconds, 2),
                          audio_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    report['status'] = 'passed'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True, indent=2))
