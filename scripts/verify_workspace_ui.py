"""Real browser checks with an isolated dictionary and a temporary test voice."""
import argparse
import io
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
import wave
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
parser=argparse.ArgumentParser()
parser.add_argument('--smoke',action='store_true')
parser.add_argument('--real-dialogue',action='store_true')
args=parser.parse_args()
output=ROOT/'data/ui-check'
output.mkdir(exist_ok=True)
report={'screenshots':[],'checks':[],'console_errors':[]}
test_speaker='ui_test_'+uuid4().hex[:12]
enrolled=False

import httpx
import uvicorn
from playwright.sync_api import sync_playwright,expect

with tempfile.TemporaryDirectory(prefix='voice-ui-verification-') as temporary:
    os.environ['DICTIONARY_PATH']=str(Path(temporary)/'dictionary')
    os.environ['JOURNAL_PATH']=str(Path(temporary)/'journal')
    from voice_id_mvp.audio import write_wav_slice
    clip=Path(temporary)/'example.wav'
    write_wav_slice(ROOT/'data/synthetic_russian_tts.wav',clip,1,4)
    with socket.socket() as listener:
        listener.bind(('127.0.0.1',0))
        port=listener.getsockname()[1]
    server=uvicorn.Server(uvicorn.Config('voice_id_mvp.services.workspace_app:app',host='127.0.0.1',port=port,log_level='error'))
    thread=threading.Thread(target=server.run,daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(.05)
    if not server.started:
        raise RuntimeError('Isolated UI server did not start')
    try:
        with sync_playwright() as playwright:
            browser=playwright.chromium.launch(channel='msedge',headless=True)
            context=browser.new_context(viewport={'width':1440,'height':1000},accept_downloads=True)
            page=context.new_page()
            page.set_default_timeout(30000)
            page.on('pageerror',lambda error:report['console_errors'].append(str(error)))
            page.on('console',lambda message:report['console_errors'].append(message.text) if message.type=='error' else None)
            page.goto(f'http://127.0.0.1:{port}/')
            expect(page.locator('#connection-status')).to_contain_text('знакомых голосов',timeout=30000)
            assert page.locator('.tab').count()==5
            page.locator('#tab-dialogue').focus()
            page.keyboard.press('ArrowDown')
            expect(page.locator('#tab-voices')).to_be_focused()
            expect(page.locator('#panel-voices')).to_be_visible()
            report['checks'].append('keyboard tabs, labels, initial voice list')

            page.locator('#speaker-name').fill(test_speaker)
            page.locator('#enroll-file').set_input_files(str(ROOT/'data/synthetic_russian_tts.wav'))
            page.locator('#enroll-form button[type=submit]').click()
            expect(page.locator('#enroll-form .job-status')).to_contain_text('сохранён',timeout=180000)
            enrolled=True
            expect(page.locator('#speakers-list')).to_contain_text(test_speaker)
            report['checks'].append('real enrollment and voice list through UI')
            page.locator('#identify-file').set_input_files(str(ROOT/'data/synthetic_russian_tts.wav'))
            page.locator('#identify-form button[type=submit]').click()
            expect(page.locator('#identify-result strong')).to_have_text(test_speaker,timeout=180000)
            with httpx.Client(timeout=30,trust_env=False) as client:
                deleted=client.delete(f'http://127.0.0.1:8001/speakers/{test_speaker}')
                deleted.raise_for_status()
            enrolled=False
            page.locator('#refresh-voices').click()
            expect(page.locator('#speakers-list')).not_to_contain_text(test_speaker)
            report['checks'].append('real identify, temporary test voice removed')
            short=io.BytesIO()
            with wave.open(short,'wb') as wav:
                wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(16000)
                wav.writeframes(b'\0\0'*3200)
            page.locator('#identify-file').set_input_files({'name':'short.wav','mimeType':'audio/wav','buffer':short.getvalue()})
            page.locator('#identify-form button[type=submit]').click()
            expect(page.locator('#identify-result strong')).to_have_text('Неопределённый участник',timeout=60000)
            expect(page.locator('#identify-result')).to_contain_text('Слишком короткая')
            report['checks'].append('short recording shown as UNKNOWN with a plain-language reason')

            page.locator('#tab-transcribe').click()
            page.locator('#transcribe-file').set_input_files(str(clip))
            page.locator('#transcribe-form button[type=submit]').click()
            expect(page.locator('#transcribe-form .job-status')).to_contain_text('Готово',timeout=180000)
            assert page.locator('#transcribe-result').inner_text().strip()
            with page.expect_download() as download:
                page.locator('#transcribe-exports').get_by_role('button',name='Сохранить JSON').click()
            exported=json.loads(Path(download.value.path()).read_text(encoding='utf-8'))
            assert 'text' in exported and 'original_text' in exported
            report['checks'].append('real transcription, valid UTF-8 JSON export')
            page.locator('#transcribe-file').set_input_files({'name':'broken.mp3','mimeType':'audio/mpeg','buffer':b'broken audio'})
            page.locator('#transcribe-form button[type=submit]').click()
            expect(page.locator('#transcribe-form .job-status')).to_contain_text('не удалось прочитать',timeout=60000)
            expect(page.locator('#transcribe-form button[type=submit]')).to_be_enabled()
            page.locator('#transcribe-file').set_input_files(str(clip))
            page.locator('#transcribe-form button[type=submit]').click()
            expect(page.locator('#transcribe-form .job-status')).to_contain_text('Готово',timeout=180000)
            report['checks'].append('bad audio error shown, retry returns a successful result')

            page.locator('#tab-dictionary').click()
            page.locator('#word-name').fill('чиабатта')
            page.locator('#word-category').fill('Выпечка')
            page.locator('#word-description').fill('Итальянский хлеб с хрустящей корочкой. Тестовая карточка в отдельной базе.')
            page.locator('#word-aliases').fill('чааббат\nчабатта')
            page.locator('#word-form button[type=submit]').click()
            expect(page.locator('#dictionary-list')).to_contain_text('чиабатта')
            expect(page.locator('#example-form')).to_be_visible()
            page.locator('#example-transcript').fill('Тестовый аудиопример для проверки плеера')
            page.locator('#example-file').set_input_files(str(clip))
            page.locator('#example-form button[type=submit]').click()
            expect(page.locator('#example-status')).to_contain_text('Пример сохранён',timeout=60000)
            page.locator('.examples summary').first.click()
            audio=page.locator('.example audio').first
            audio.evaluate('(a) => a.play()')
            assert audio.evaluate('(a) => a.readyState')>=2
            audio.evaluate('(a) => a.pause()')
            page.locator('#dictionary-search').fill('неттакогослова')
            expect(page.locator('#dictionary-list')).to_contain_text('Ничего не найдено')
            page.locator('#dictionary-search').fill('')
            report['checks'].append('isolated dictionary create, search, upload and audio playback')

            page.locator('#new-word').click()
            page.locator('#word-name').fill('<img src=x onerror="window.__xss=true">')
            page.locator('#word-form button[type=submit]').click()
            expect(page.locator('#dictionary-list')).to_contain_text('<img src=x')
            assert page.locator('#dictionary-list img').count()==0
            assert page.evaluate('Boolean(window.__xss)') is False
            bad_id=page.request.get(f'http://127.0.0.1:{port}/api/dictionary').json()['words']
            bad_id=next(w['id'] for w in bad_id if '<img' in w['word'])
            assert page.request.delete(f'http://127.0.0.1:{port}/api/dictionary/{bad_id}').status==200
            page.locator('#new-word').click()
            page.locator('#tab-dictionary').click()
            report['checks'].append('HTML text not executed, archived entry no longer listed')

            if args.real_dialogue:
                source=Path('C:/Users/PC/Downloads/test4sound.mp3')
                if not source.exists():
                    raise RuntimeError('Real dialogue file is missing')
                page.locator('#tab-dialogue').click()
                page.locator('#dialogue-file').set_input_files(str(source))
                started=time.monotonic()
                page.locator('#dialogue-form button[type=submit]').click()
                expect(page.locator('#dialogue-form button[type=submit]')).to_be_disabled()
                expect(page.locator('#dialogue-form .job-status')).to_contain_text('Готово',timeout=300000)
                report['dialogue_seconds']=time.monotonic()-started
                text=page.locator('#dialogue-result').inner_text()
                assert 'Круассан вам подогреть?' in text and 'Да, если можно, было бы здорово.' in text
                turns=page.locator('.turn')
                question=next(turn for turn in turns.all() if 'Круассан вам подогреть?' in turn.inner_text())
                answer=next(turn for turn in turns.all() if 'Да, если можно, было бы здорово.' in turn.inner_text())
                assert question.locator('.speaker-chip').inner_text()=='Платон'
                assert answer.locator('.speaker-chip').inner_text()=='Прохор'
                assert 'Чиабатта' in text
                assert page.locator('.original-text').count()>0
                question.locator('.turn-time').click()
                assert page.locator('#dialogue-form audio').evaluate('(a) => a.currentTime')>=10
                with page.expect_download() as download:
                    page.locator('#dialogue-exports').get_by_role('button',name='Сохранить JSON').click()
                data=json.loads(Path(download.value.path()).read_text(encoding='utf-8'))
                assert data['dictionary_corrections']
                (output/'real-dialogue.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
                report['checks'].append('real test4: Platon question / Prokhor answer, dictionary correction, original text, timeline playback and JSON')

            page.locator('#tab-journal').click()
            expect(page.locator('#journal-list')).to_contain_text('broken.mp3')
            page.locator('#journal-list').get_by_role('button',name='broken.mp3',exact=True).click()
            expect(page.locator('#journal-result')).to_contain_text('не удалось прочитать')
            assert not page.locator('#journal-player').is_hidden()
            with page.expect_download() as download:
                page.locator('#journal-actions').get_by_text('Скачать аудио',exact=True).click()
            assert Path(download.value.path()).read_bytes()==b'broken audio'
            page.once('dialog',lambda dialog:dialog.accept())
            page.locator('#journal-actions').get_by_role('button',name='Удалить запись',exact=True).click()
            expect(page.locator('#journal-result')).to_contain_text('удалены')
            expect(page.locator('#journal-list')).not_to_contain_text('broken.mp3')
            report['checks'].append('failed recording audio retained/downloadable; explicit confirmed deletion')
            page.reload()
            page.locator('#tab-journal').click()
            selected_name='test4sound.mp3' if args.real_dialogue else 'example.wav'
            page.locator('#journal-list').get_by_role('button',name=selected_name,exact=True).first.click()
            expect(page.locator('#journal-exports')).to_be_visible()
            if args.real_dialogue:
                expect(page.locator('#journal-result')).to_contain_text('Круассан вам подогреть?')
                page.locator('#journal-result .turn-time').first.click()
            else:
                page.locator('#journal-player').evaluate('(a)=>a.play()')
            page.locator('#journal-player').evaluate('(a)=>a.pause()')
            with page.expect_download() as download:
                page.locator('#journal-actions').get_by_role('button',name='Сохранить запись JSON',exact=True).click()
            journal_export=json.loads(Path(download.value.path()).read_text(encoding='utf-8'))
            assert journal_export['result'] and journal_export['filename']==selected_name
            page.locator('#journal-search').fill('неттакогодиалога')
            expect(page.locator('#journal-list')).to_contain_text('Записей нет')
            page.locator('#journal-search').fill('')
            expect(page.locator('#journal-list')).to_contain_text(selected_name)
            report['checks'].append('journal survives reload; readable result, playback, JSON metadata and text search')

            for width in [1440,390]:
                page.set_viewport_size({'width':width,'height':1000})
                for tab in ['dialogue','voices','transcribe','dictionary','journal']:
                    page.locator('#tab-'+tab).click()
                    page.wait_for_timeout(350)
                    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'),f'Overflow at {width}/{tab}'
                    screenshot=output/f'{tab}-{width}.png'
                    page.screenshot(path=str(screenshot),full_page=True)
                    report['screenshots'].append(str(screenshot))
            assert not report['console_errors'],report['console_errors']
            report['checks'].append('10 screenshots, no horizontal overflow at 1440/390, no console errors')
            browser.close()
    finally:
        if enrolled:
            with httpx.Client(timeout=30,trust_env=False) as client:
                client.delete(f'http://127.0.0.1:8001/speakers/{test_speaker}').raise_for_status()
        server.should_exit=True
        thread.join(30)
    report['status']='passed'
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True,indent=2))
