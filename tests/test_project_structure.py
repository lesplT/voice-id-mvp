"""Guard the requested package boundaries, explicit imports, and documentation style."""
import ast
from pathlib import Path

import yaml

from voice_id_mvp.core.config import ROOT, Settings


def test_project_root_and_storage_paths_survive_module_move():
    assert ROOT == Path(__file__).resolve().parents[1]
    settings = Settings(_env_file=None, journal_path='data/journal', dictionary_path='data/dictionary')
    assert settings.journal_path == ROOT / 'data/journal'
    assert settings.dictionary_path == ROOT / 'data/dictionary'
    assert Settings.model_config['env_file'] == ROOT / '.env'


def test_application_root_contains_only_package_initializer():
    assert {p.name for p in (ROOT / 'voice_id_mvp').glob('*.py')} == {'__init__.py'}
    for name in ('core', 'audio', 'recognition', 'application', 'storage', 'services'):
        assert (ROOT / 'voice_id_mvp' / name / '__init__.py').is_file()


def test_python_imports_are_absolute_and_explicit():
    for folder in ('voice_id_mvp', 'scripts', 'tests'):
        for path in (ROOT / folder).rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                if isinstance(node, ast.ImportFrom):
                    assert node.level == 0, f'Relative import in {path}:{node.lineno}'
                    assert all(item.name != '*' for item in node.names), f'Wildcard import in {path}'


def test_production_classes_and_functions_have_english_contracts():
    documented = 0
    for path in (ROOT / 'voice_id_mvp').rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node) or ''
                for section in ('Purpose:', 'Ingress:', 'Egress:'):
                    assert section in doc, f'Missing {section} in {path}:{node.name}'
                assert not any('\u0400' <= char <= '\u04ff' for char in doc), f'Non-English contract: {path}:{node.name}'
                documented += 1
    assert documented >= 125


def test_single_compose_preserves_runtime_identity_and_data():
    assert not (ROOT / 'compose.docker.yml').exists()
    config = yaml.safe_load((ROOT / 'docker-compose.yml').read_text(encoding='utf-8'))
    assert config['name'] == 'voice-id-mvp-docker'
    services = config['services']
    assert services['qdrant']['volumes'] == ['./data/docker/qdrant:/qdrant/storage']
    assert 'ports' not in services['qdrant']
    for name in ('workspace', 'transcription'):
        assert './data/docker/workspace:/app/data' in services[name]['volumes']
    native = services['qdrant-native']
    assert native['profiles'] == ['native']
    assert native['ports'] == ['127.0.0.1:6333:6333']
    assert native['volumes'] == ['./data/qdrant_storage:/qdrant/storage']


def test_native_scripts_do_not_stop_the_full_stack_database():
    scripts = ROOT / 'scripts'
    if not scripts.exists():
        return
    assert 'docker compose --profile native up -d qdrant-native' in (scripts / 'start_all.ps1').read_text()
    assert 'docker compose --profile native stop qdrant-native' in (scripts / 'stop_all.ps1').read_text()
