"""Exact, auditable text substitutions. Never guesses words or speaker identity."""
from copy import deepcopy
import re
import unicodedata

from .dictionary import DictionaryStore, word_key


def correct_text(text: str,entries: list[dict]):
    mapping={}
    for entry in entries:
        if not entry.get('active',True) or entry.get('archived',False):
            continue
        for alias in entry.get('aliases',[]):
            key=word_key(alias)
            if key and key != word_key(entry['word']):
                previous=mapping.get(key)
                if previous and previous['word'] != entry['word']:
                    raise ValueError('Конфликт вариантов словаря: '+alias)
                mapping[key]=entry
    if not mapping:
        return text,[]
    pattern=re.compile(r'(?<!\w)(?:'+ '|'.join(re.escape(key) for key in sorted(mapping,key=len,reverse=True))+r')(?!\w)',re.IGNORECASE)
    changes=[]
    def substitute(match):
        before=match.group()
        entry=mapping.get(word_key(before))
        if entry is None:
            return before
        after=entry['word']
        if before.isupper():
            after=after.upper()
        elif before[0].isupper():
            after=after[0].upper()+after[1:]
        changes.append({'word_id':entry['id'],'before':before,'after':after})
        return after
    return pattern.sub(substitute,unicodedata.normalize('NFC',text)),changes


def apply_dictionary_result(result: dict, enabled: bool=True, entries: list[dict] | None=None):
    output=deepcopy(result)
    # Prevent double application when forwarding an already-decorated response.
    if 'dictionary_enabled' in output:
        return output
    entries=(DictionaryStore().list() if entries is None else entries) if enabled else []
    output['dictionary_enabled']=enabled
    output['original_text']=output.get('text','')
    if 'segments' in output:
        all_changes=[]
        for segment in output['segments']:
            original=segment.get('text','')
            segment['original_text']=original
            segment['text'],segment['dictionary_corrections']=correct_text(original,entries)
            all_changes.extend(segment['dictionary_corrections'])
        output['text']='\n'.join(f"[{s['start']:.2f}–{s['end']:.2f}] {s['speaker_id']}: {s['text']}" for s in output['segments'])
        output['dictionary_corrections']=all_changes
    else:
        output['text'],output['dictionary_corrections']=correct_text(output.get('text',''),entries)
    return output
