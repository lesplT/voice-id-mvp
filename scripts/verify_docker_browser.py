"""Live Docker UI checks; removes only its own synthetic test profile/recording."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from uuid import uuid4

from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from voice_id_mvp.audio import write_wav_slice

output = ROOT / 'data/docker-check'
output.mkdir(exist_ok=True)
speaker = 'docker_test_' + uuid4().hex[:12]
enrolled = False
recording = None
report = {'checks': [], 'console_errors': []}
with tempfile.TemporaryDirectory(prefix='docker-ui-check-') as temporary:
    clip = Path(temporary) / 'example.wav'
    write_wav_slice(ROOT / 'data/synthetic_russian_tts.wav', clip, 1, 4)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page(viewport={'width':1440, 'height':1000}, accept_downloads=True)
        page.on('pageerror', lambda error:report['console_errors'].append(str(error)))
        try:
            page.goto('http://127.0.0.1:18000/')
            expect(page.locator('#connection-status')).to_contain_text('знакомых голосов: 2')
            page.locator('#tab-voices').click()
            page.locator('#speaker-name').fill(speaker)
            page.locator('#enroll-file').set_input_files(str(clip))
            page.locator('#enroll-form button[type=submit]').click()
            expect(page.locator('#enroll-form .job-status')).to_contain_text('сохранён', timeout=120000)
            enrolled = True
            page.locator('#identify-file').set_input_files(str(clip))
            page.locator('#identify-form button[type=submit]').click()
            expect(page.locator('#identify-result strong')).to_have_text(speaker, timeout=120000)
            report['checks'].append('real container enrollment and identification through UI')

            page.locator('#tab-transcribe').click()
            page.locator('#transcribe-file').set_input_files(str(clip))
            page.locator('#transcribe-form button[type=submit]').click()
            expect(page.locator('#transcribe-form .job-status')).to_contain_text('Готово', timeout=120000)
            with page.expect_download() as download:
                page.locator('#transcribe-exports').get_by_role('button', name='Сохранить JSON').click()
            result = json.loads(Path(download.value.path()).read_text(encoding='utf-8'))
            recording = result['journal_id']
            assert result['engine'] == 'gigaam' and result['text'].strip()
            report['checks'].append('real GigaAM v3 transcription and JSON export through Docker UI')

            saved = json.loads((output / 'live-report.json').read_text(encoding='utf-8'))
            page.locator('#tab-journal').click()
            # Open the verified real conversation, not the synthetic transcription.
            page.request.delete('http://127.0.0.1:18000/api/journal/' + recording)
            recording = None
            page.evaluate('(id)=>openRecording(id)', saved['recording_id'])
            expect(page.locator('#journal-result')).to_contain_text('Круассан вам подогреть?', timeout=30000)
            page.locator('#journal-result .turn-time').first.click()
            page.locator('#journal-player').evaluate('(a)=>a.play()')
            assert page.locator('#journal-player').evaluate('(a)=>a.readyState') >= 2
            page.locator('#journal-player').evaluate('(a)=>a.pause()')
            page.locator('#refresh-journal').click()
            report['checks'].append('real dialogue journal opens and plays stored audio in Docker UI')
            for width in (1440, 390):
                page.set_viewport_size({'width':width, 'height':1000})
                assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
                page.screenshot(path=str(output / f'docker-journal-{width}.png'), full_page=True)
            assert not report['console_errors']
            report['checks'].append('desktop/mobile journal screenshots; no page errors or horizontal overflow')
        finally:
            if recording:
                page.request.delete('http://127.0.0.1:18000/api/journal/' + recording)
            if enrolled:
                command = f"import urllib.request; urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8001/speakers/{speaker}', method='DELETE'))"
                subprocess.run(['docker','compose','-f','compose.docker.yml','exec','-T','enrollment','python','-c',command], cwd=ROOT, check=True)
            browser.close()
report['status'] = 'passed'
(output / 'browser-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=True, indent=2))
