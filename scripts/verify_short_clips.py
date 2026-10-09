import sys, json, tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import httpx
from voice_id_mvp.audio import write_wav_slice, normalized_upload
ROOT=Path(__file__).resolve().parents[1]
rows=[]
with tempfile.TemporaryDirectory(prefix='short-voice-check-') as tmp, httpx.Client(timeout=120) as client:
    destination=Path(tmp)/'probe.wav'
    def check(source,start,duration,expected):
        write_wav_slice(source,destination,start,start+duration)
        with destination.open('rb') as audio:
            r=client.post('http://127.0.0.1:8002/identify',files={'audio':('probe.wav',audio,'audio/wav')})
        r.raise_for_status()
        rows.append({'start':start,'duration':duration,'expected':expected,**r.json()})
    source=ROOT/'data/short-evaluation/dialogue.wav'
    for duration in [.35,.5,1.0]:
        for name,starts in [('platon',[14.35,14.45,14.6]),('prokhor',[15.95,16.3,16.7])]:
            for start in starts:
                check(source,start,duration,name)
    check(source,14.35,.2,'UNKNOWN')
    with normalized_upload((ROOT/'data/synthetic_russian_tts.wav').read_bytes(),'.wav') as synthetic:
        check(synthetic,1,2,'UNKNOWN')
    for duration in [.35,.5,1.0]:
        group=[r for r in rows if r['duration']==duration and r['expected']!='UNKNOWN']
        print(duration,{'accepted_correct':sum(r['speaker_id']==r['expected'] for r in group),
            'unknown':sum(r['speaker_id']=='UNKNOWN' for r in group),
            'wrong_known':sum(r['speaker_id'] not in [r['expected'],'UNKNOWN'] for r in group)},flush=True)
    print('negative checks',[(r['duration'],r['speaker_id'],r.get('reason')) for r in rows if r['expected']=='UNKNOWN'])
(ROOT/'data/short-evaluation/short-clips.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
