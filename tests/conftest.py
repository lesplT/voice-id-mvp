import pytest


@pytest.fixture(autouse=True)
def isolated_journal(monkeypatch, tmp_path):
    """Tests must never recover or mutate the user's production journal."""
    from voice_id_mvp.journal import JournalStore
    from voice_id_mvp.services import workspace_app
    monkeypatch.setattr(workspace_app, 'journal_store', lambda: JournalStore(tmp_path / 'journal'))
