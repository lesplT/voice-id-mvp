import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json, urllib.request, wave
import numpy as np
import torch, torchaudio
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['soundfile']
from speechbrain.lobes.features import Fbank
from speechbrain.processing.features import InputNormalization
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(4)
folder = ROOT / 'models/ecapa'
folder.mkdir(exist_ok=True)
checkpoint = folder / 'embedding_model.ckpt'
if not checkpoint.exists():
    urllib.request.urlretrieve('https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/main/embedding_model.ckpt', checkpoint.with_suffix('.part'))
    checkpoint.with_suffix('.part').replace(checkpoint)
features = Fbank(n_mels=80).eval()
norm = InputNormalization(norm_type='sentence', std_norm=False).eval()
model = ECAPA_TDNN(input_size=80, channels=[1024,1024,1024,1024,3072], kernel_sizes=[5,3,3,3,1], dilations=[1,2,3,4,1], attention_channels=128, lin_neurons=192).eval()
model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
def load(name):
    with wave.open(str(ROOT / f'data/short-evaluation/{name}.wav')) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(np.float32)/32768
def embed(audio):
    with torch.inference_mode():
        x=torch.from_numpy(audio.copy()).unsqueeze(0)
        lengths=torch.ones(1)
        v=model(norm(features(x),lengths),lengths).reshape(-1).numpy()
        return v/np.linalg.norm(v)
refs={name:embed(load(name)) for name in ['platon','prokhor']}
audio=load('dialogue')
baseline=json.loads((ROOT/'data/short-evaluation/vosk-baseline.json').read_text())
print('BASELINE TYPE', type(baseline), flush=True)
rows=baseline if isinstance(baseline,list) else baseline.get('segments',[])
results=[]
for row in rows:
    start,end=row['start'],row['end']
    v=embed(audio[int(start*16000):int(end*16000)])
    result={'start':start,'end':end,**{k:float(v@r) for k,r in refs.items()}}
    results.append(result);print(result,flush=True)
(ROOT/'data/short-evaluation/ecapa-baseline.json').write_text(json.dumps(results,indent=2))
for start in np.arange(14.2,18.1,0.2):
    v=embed(audio[int(start*16000):int((start+.8)*16000)])
    print('WINDOW',round(start,2),{k:round(float(v@r),3) for k,r in refs.items()},flush=True)
