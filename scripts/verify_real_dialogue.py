"""Local real-audio acceptance check. Does not upload audio externally."""
import sys, json, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import httpx
ROOT = Path(__file__).resolve().parents[1]
files = Path('C:/Users/PC/Downloads')
parser=argparse.ArgumentParser()
parser.add_argument('--skip-enrollment',action='store_true')
args=parser.parse_args()
with httpx.Client(timeout=600) as client:
    enrollment = []
    for name, filename in ([] if args.skip_enrollment else [('platon','test1sound.mp3'),('prokhor','test3sound.mp3')]):
        with (files/filename).open('rb') as audio:
            r=client.post(f'http://127.0.0.1:8001/speakers/{name}/enroll', files={'audio':(filename,audio,'audio/mpeg')})
        r.raise_for_status(); enrollment.append(r.json()); print(r.json(),flush=True)
    start=time.monotonic()
    with (files/'test4sound.mp3').open('rb') as audio:
        r=client.post('http://127.0.0.1:8003/dialogue',files={'audio':('test4sound.mp3',audio,'audio/mpeg')})
    r.raise_for_status()
    result={'elapsed_seconds':time.monotonic()-start,'enrollment':enrollment,'dialogue':r.json()}
    (ROOT/'data/short-evaluation/ecapa-dialogue.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
