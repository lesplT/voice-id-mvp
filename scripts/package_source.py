"""Package code only: explicitly excludes model weights and personal recordings."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import argparse

ROOT = Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('destination', type=Path)
destination=parser.parse_args().destination.resolve()
destination.parent.mkdir(parents=True,exist_ok=True)
with ZipFile(destination,'w',compression=ZIP_DEFLATED) as archive:
    for name in ['voice_id_mvp','scripts','tests','.github']:
        for path in (ROOT/name).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                archive.write(path,path.relative_to(ROOT))
    for path in ROOT.iterdir():
        if path.is_file() and path.name != '.env' and path.suffix != '.zip':
            archive.write(path,path.name)
print(destination)
