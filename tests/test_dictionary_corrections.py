import pytest
from voice_id_mvp.dictionary_corrections import correct_text,apply_dictionary_result

WORDS=[{'id':'one','word':'чиабатта','aliases':['чааббат','чабатта'],'active':True}]


def test_exact_boundaries_case_and_numbers():
    result,changes=correct_text('Чааббат! ЧААББАТ и чааббатная, +7 999 123 4567.',WORDS)
    assert result=='Чиабатта! ЧИАБАТТА и чааббатная, +7 999 123 4567.'
    assert len(changes)==2
    assert correct_text(result,WORDS)[0]==result
    assert correct_text('неизвестная выпечка',WORDS)[0]=='неизвестная выпечка'


def test_disabled_preserves_text():
    assert correct_text('чааббат',[{**WORDS[0],'active':False}])==('чааббат',[])
    result=apply_dictionary_result({'text':'чааббат'},enabled=False,entries=WORDS)
    assert result['text']==result['original_text']=='чааббат'
    assert result['dictionary_corrections']==[]


def test_dialogue_keeps_voice_and_time():
    source={'text':'original','segments':[{'start':1,'end':2,'speaker_id':'platon','score':.6,'text':'чааббат'}]}
    result=apply_dictionary_result(source,entries=WORDS)
    segment=result['segments'][0]
    assert segment['speaker_id']=='platon' and segment['score']==.6
    assert segment['start']==1 and segment['end']==2
    assert segment['text']=='чиабатта' and segment['original_text']=='чааббат'
    assert source['segments'][0]['text']=='чааббат'
    assert apply_dictionary_result(result,entries=WORDS)==result


def test_phrase_precedence_and_unicode():
    words=[{'id':'x','word':'батон','aliases':['белый хлеб'],'active':True}, {'id':'y','word':'хлебушек','aliases':['хлеб'],'active':True}]
    assert correct_text('белый хлеб и хлеб',words)[0]=='батон и хлебушек'
    assert correct_text('чааббат',WORDS)[0]=='чиабатта'
    with pytest.raises(ValueError):
        correct_text('чааббат',WORDS+[{'id':'other','word':'не то','aliases':['чааббат']}])
