'use strict';
const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
let knownSpeakers = [];
let dictionaryWords = [];
let selectedWord = null;
let selectedRecording = null;
let journalOffset = 0;
let journalRequest = 0;
const results = new Map();
const previewUrls = new Map();
const busyForms = new Set();
const reasons = {
  too_short: 'Слишком короткая запись: нужно хотя бы 0,35 секунды.',
  below_threshold: 'Голос недостаточно похож на сохранённые образцы.',
  ambiguous: 'Есть несколько похожих голосов — уверенно выбрать одного не получилось.',
  no_embedding: 'Не удалось выделить признаки голоса. Попробуйте запись с более отчётливой речью.',
  no_speech: 'Речь не найдена. Возможно, в записи тишина или только шум.'
};

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}
function button(text, action, className = 'button small-button') {
  const node = element('button', text, className);
  node.type = 'button'; node.addEventListener('click', action);
  return node;
}
function friendlyName(id) {
  return { platon: 'Платон', prokhor: 'Прохор', UNKNOWN: 'Неопределённый участник' }[id] || id;
}
function timeLabel(seconds) {
  const value = Math.max(0, Number(seconds) || 0);
  return `${Math.floor(value / 60)}:${String(Math.floor(value % 60)).padStart(2, '0')}`;
}
function plural(count, forms) {
  const key = new Intl.PluralRules('ru').select(count);
  return forms[key === 'one' ? 0 : key === 'few' ? 1 : 2];
}
function notify(text) {
  const toast = $('#toast'); toast.textContent = text; toast.hidden = false;
  clearTimeout(notify.timer); notify.timer = setTimeout(() => { toast.hidden = true; }, 4000);
}
function status(node, text, error = false) {
  node.hidden = false; node.textContent = text;
  node.classList.toggle('error', error);
}
async function api(path, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 240000);
  try {
    const response = await fetch(path, { ...options, signal: controller.signal });
    const text = await response.text();
    let data;
    try { data = JSON.parse(text); } catch { throw new Error('Сервис вернул непонятный ответ. Обновите страницу и повторите.'); }
    if (!response.ok) {
      const detail = data.detail;
      throw new Error(typeof detail === 'string' ? detail : 'Проверьте заполнение полей и выбранный файл.');
    }
    return data;
  } catch (error) {
    if (error.name === 'AbortError') throw new Error('Ответ слишком долго не приходит. Проверьте, запущены ли сервисы.');
    if (error instanceof TypeError) throw new Error('Нет связи с сервисом. Запустите проект и повторите.');
    throw error;
  } finally { clearTimeout(timeout); }
}

function activateTab(id, focus = false) {
  for (const tab of $$('.tab')) {
    const active = tab.dataset.tab === id;
    tab.classList.toggle('active', active); tab.setAttribute('aria-selected', String(active));
    tab.tabIndex = active ? 0 : -1;
    $(`#panel-${tab.dataset.tab}`).hidden = !active;
    if (active && focus) tab.focus();
  }
  if (id === 'dictionary') loadDictionary();
  if (id === 'voices') loadSpeakers();
  if (id === 'journal') loadJournal();
}
for (const tab of $$('.tab')) {
  tab.addEventListener('click', () => activateTab(tab.dataset.tab));
  tab.addEventListener('keydown', (event) => {
    const tabs = $$('.tab'); const i = tabs.indexOf(tab);
    let next;
    if (['ArrowDown','ArrowRight'].includes(event.key)) next = (i + 1) % tabs.length;
    if (['ArrowUp','ArrowLeft'].includes(event.key)) next = (i + tabs.length - 1) % tabs.length;
    if (event.key === 'Home') next = 0;
    if (event.key === 'End') next = tabs.length - 1;
    if (next !== undefined) { event.preventDefault(); activateTab(tabs[next].dataset.tab, true); }
  });
}

function previewFile(form) {
  const file = $('input[type=file]', form).files[0];
  const preview = $('.file-preview', form);
  if (previewUrls.has(form.id)) URL.revokeObjectURL(previewUrls.get(form.id));
  previewUrls.delete(form.id);
  if (!file) { preview.hidden = true; return; }
  if (file.size > 100 * 1024 * 1024) {
    status($('.job-status', form), 'Файл слишком большой. Максимум — 100 МБ.', true);
    $('input[type=file]', form).value = ''; preview.hidden = true; return;
  }
  const url = URL.createObjectURL(file); previewUrls.set(form.id, url);
  $('.file-info', form).textContent = `${file.name} · ${(file.size / (1024 * 1024)).toFixed(1)} МБ`;
  $('audio', preview).src = url; preview.hidden = false;
}
for (const form of $$('.audio-form')) {
  const input = $('input[type=file]', form);
  input.addEventListener('change', () => previewFile(form));
  const zone = $('.upload-zone', form);
  for (const eventName of ['dragenter', 'dragover']) zone.addEventListener(eventName, (event) => {
    event.preventDefault(); zone.classList.add('dragover');
  });
  for (const eventName of ['dragleave', 'drop']) zone.addEventListener(eventName, (event) => {
    event.preventDefault(); zone.classList.remove('dragover');
  });
  zone.addEventListener('drop', (event) => {
    if (event.dataTransfer.files.length && !busyForms.has(form.id)) {
      const transfer = new DataTransfer(); transfer.items.add(event.dataTransfer.files[0]);
      input.files = transfer.files; previewFile(form);
    }
  });
  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    if (busyForms.has(form.id)) return;
    const file = input.files[0]; if (!file) return;
    const operation = form.dataset.operation;
    const node = $('.job-status', form);
    busyForms.add(form.id);
    const controls = $$('input,button', form);
    const data = new FormData(); data.append('operation', operation); data.append('audio', file);
    data.append('use_dictionary', String(Boolean($('input[name=use_dictionary]', form)?.checked)));
    if (operation === 'enroll') {
      const entered = $('#speaker-name').value.trim();
      const existing = knownSpeakers.find(item => friendlyName(item.speaker_id).toLocaleLowerCase('ru') === entered.toLocaleLowerCase('ru'));
      data.append('speaker_id', existing ? existing.speaker_id : entered);
    }
    controls.forEach(control => { control.disabled = true; });
    status(node, 'Загружаем запись…');
    const started = Date.now();
    try {
      let job = await api('/api/jobs', { method: 'POST', body: data });
      while (!['done','failed'].includes(job.status)) {
        const elapsed = Math.floor((Date.now() - started) / 1000);
        status(node, job.status === 'queued' ? `Ожидаем в очереди · ${elapsed} с` : `Обрабатываем запись · ${elapsed} с. На первой записи загрузка модели может занять время.`);
        await new Promise(resolve => setTimeout(resolve, 900));
        job = await api(`/api/jobs/${job.id}`);
      }
      if (job.status === 'failed') throw new Error(job.error || 'Не удалось обработать запись.');
      results.set(operation, job.result);
      status(node, `Готово · ${Math.round((Date.now() - started) / 1000)} с`);
      if (operation === 'dialogue') renderDialogue(job.result);
      if (operation === 'transcribe') renderTranscript(job.result);
      if (job.result?.journal_id) {
        status(node, `Готово · ${Math.round((Date.now() - started) / 1000)} с. Запись и результат сохранены в «Журнале».`);
      }
      if (operation === 'identify') renderIdentity(job.result);
      if (operation === 'enroll') {
        status(node, `Голос «${friendlyName(job.result.speaker_id)}» сохранён. Теперь его можно узнать в диалоге.`);
        await loadSpeakers(); notify('Голос сохранён');
      }
    } catch (error) { status(node, error.message + (['dialogue','transcribe'].includes(operation) ? ' Если запись принята сервером, её исходное аудио доступно в «Журнале».' : ''), true); }
    finally {
      busyForms.delete(form.id); controls.forEach(control => { control.disabled = false; });
    }
  });
}

function download(filename, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = element('a'); link.href = url; link.download = filename;
  document.body.append(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function resultText(operation, result) {
  if (operation === 'dialogue') return (result.segments || []).map(item => `[${timeLabel(item.start)}–${timeLabel(item.end)}] ${friendlyName(item.speaker_id)}: ${item.text || ''}`).join('\n');
  return result.text || '';
}
function renderExports(operation, result, view = operation) {
  const bar = $(`#${view}-exports`); bar.replaceChildren(); bar.hidden = false;
  bar.append(button('Копировать текст', async () => {
    try { await navigator.clipboard.writeText(resultText(operation, result)); notify('Текст скопирован'); }
    catch { notify('Не удалось скопировать. Сохраните текст кнопкой TXT.'); }
  }));
  bar.append(button('Сохранить TXT', () => download(`voice-id-${operation}.txt`, resultText(operation, result), 'text/plain;charset=utf-8')));
  bar.append(button('Сохранить JSON', () => download(`voice-id-${operation}.json`, JSON.stringify(result, null, 2), 'application/json;charset=utf-8')));
}
function showOriginal(container, original) {
  if (!original) return;
  const details = element('details', undefined, 'original-text');
  details.append(element('summary', 'Показать исходный текст до словаря'), element('p', original));
  container.append(details);
}
function renderDialogue(result, view = 'dialogue', playerSelector = '#dialogue-form audio') {
  const container = $(`#${view}-result`); container.replaceChildren();
  const segments = result.segments || [];
  if (view === 'dialogue') $('#dialogue-count').textContent = `${segments.length} ${plural(segments.length, ['реплика','реплики','реплик'])}`;
  renderExports('dialogue', result, view);
  if (!segments.length) {
    container.append(element('p', 'В записи не найдена речь. Попробуйте другую запись.', 'muted')); return;
  }
  const colors = new Map();
  for (const segment of segments) {
    if (!colors.has(segment.speaker_id)) colors.set(segment.speaker_id, colors.size % 3);
    const turn = element('article', undefined, 'turn');
    const header = element('div', undefined, 'turn-header');
    header.append(element('span', friendlyName(segment.speaker_id), `speaker-chip ${segment.speaker_id === 'UNKNOWN' ? 'unknown' : `speaker-${colors.get(segment.speaker_id)}`}`));
    header.append(button(`${timeLabel(segment.start)} – ${timeLabel(segment.end)} ↗`, async () => {
      const player = $(playerSelector);
      if (!player.src) return;
      player.currentTime = Number(segment.start);
      try { await player.play(); } catch { notify('Нажмите кнопку воспроизведения на плеере.'); }
    }, 'turn-time'));
    turn.append(header, element('p', segment.text || 'Нет распознанного текста.'));
    if (segment.speaker_id === 'UNKNOWN') turn.append(element('div', reasons[segment.reason] || 'Недостаточно данных, чтобы уверенно узнать голос.', 'turn-reason'));
    if (segment.dictionary_corrections?.length) turn.append(element('div', `Исправлено по словарю: ${segment.dictionary_corrections.length}`, 'turn-reason'));
    if (segment.original_text && segment.original_text !== segment.text) showOriginal(turn, segment.original_text);
    container.append(turn);
  }
}
function renderTranscript(result, view = 'transcribe') {
  const container = $(`#${view}-result`); container.replaceChildren();
  renderExports('transcribe', result, view);
  if (result.dictionary_corrections?.length) container.append(element('div', `Исправлений по словарю: ${result.dictionary_corrections.length}`, 'correction-badge'));
  container.append(element('div', result.text || 'В записи не удалось распознать текст.', 'transcript-text'));
  if (result.original_text && result.original_text !== result.text) showOriginal(container, result.original_text);
}
function renderIdentity(result) {
  const node = $('#identify-result'); node.replaceChildren();
  node.append(element('strong', friendlyName(result.speaker_id)));
  if (result.reason) node.append(element('p', reasons[result.reason] || 'Говорящий не определён.', 'muted'));
  node.append(button('Сохранить JSON', () => download('voice-id-identify.json', JSON.stringify(result, null, 2), 'application/json;charset=utf-8')));
}
async function loadSpeakers() {
  const node = $('#speakers-list');
  try {
    knownSpeakers = (await api('/api/speakers')).speakers || [];
    node.replaceChildren();
    if (!knownSpeakers.length) node.append(element('p', 'Пока нет знакомых голосов. Добавьте первый образец слева.', 'muted'));
    for (const person of knownSpeakers) {
      const row = element('div', undefined, 'speaker-row');
      row.append(element('span', friendlyName(person.speaker_id).slice(0, 1).toLocaleUpperCase('ru'), 'avatar'));
      const info = element('div'); info.append(element('strong', friendlyName(person.speaker_id)), element('small', `Сохранённых записей: ${person.enrollments} · образцов голоса: ${person.references}`));
      row.append(info); node.append(row);
    }
    $('#connection-status').textContent = `Локально · знакомых голосов: ${knownSpeakers.length}`;
    $('#connection-status').classList.remove('offline');
  } catch (error) {
    node.textContent = error.message;
    $('#connection-status').textContent = 'Сервис голосов недоступен';
    $('#connection-status').classList.add('offline');
  }
}
$('#refresh-voices').addEventListener('click', loadSpeakers);

function editWord(word) {
  selectedWord = word;
  $('#word-form-title').textContent = word ? 'Редактировать слово' : 'Добавить слово';
  $('#word-name').value = word?.word || '';
  $('#word-category').value = word?.category || '';
  $('#word-description').value = word?.description || '';
  $('#word-aliases').value = word?.aliases.join('\n') || '';
  $('#word-active').checked = word ? word.active : true;
  $('#word-status').hidden = true; $('#example-status').hidden = true;
  $('#example-form').hidden = !word;
  $('#example-word').textContent = word ? `Для слова «${word.word}»` : '';
  $('#example-transcript').value = word?.word || '';
  $('#example-file').value = '';
  $('#word-name').focus();
}
$('#new-word').addEventListener('click', () => editWord(null));
async function loadDictionary() {
  const node = $('#dictionary-list');
  try {
    dictionaryWords = (await api('/api/dictionary')).words;
    renderDictionary();
  } catch (error) { node.textContent = error.message; }
}
function renderDictionary() {
  const query = $('#dictionary-search').value.trim().toLocaleLowerCase('ru');
  const words = dictionaryWords.filter(word => `${word.word} ${word.category} ${word.description}`.toLocaleLowerCase('ru').includes(query));
  $('#dictionary-count').textContent = `${dictionaryWords.length} ${plural(dictionaryWords.length, ['слово','слова','слов'])}`;
  const list = $('#dictionary-list'); list.replaceChildren();
  if (!words.length) {
    const empty = element('div', undefined, 'empty-state');
    empty.append(element('span', '▤', 'empty-icon'), element('h3', query ? 'Ничего не найдено' : 'Ваш словарь пока пуст'), element('p', query ? 'Попробуйте другое слово или категорию.' : 'Добавьте слово и варианты ошибок. Затем можно прикрепить аудиопримеры произношения.'));
    list.append(empty);
  }
  for (const word of words) {
    const card = element('article', undefined, 'word-card');
    const title = element('div', undefined, 'word-title'); title.append(element('h3', word.word));
    if (word.category) title.append(element('span', word.category, 'tag'));
    title.append(element('span', word.active ? 'Исправления включены' : 'Исправления выключены', `tag ${word.active ? '' : 'disabled'}`));
    card.append(title);
    if (word.description) card.append(element('p', word.description));
    card.append(element('p', word.aliases.length ? `Исправляем: ${word.aliases.join(', ')} → ${word.word}` : 'Варианты ошибок не заданы. Слово и примеры сохраняются, текст не изменяется.'));
    const actions = element('div', undefined, 'word-actions');
    actions.append(button('Изменить / добавить пример', () => editWord(word)), button('В архив', async () => {
      if (!window.confirm(`Убрать «${word.word}» из словаря? Записи сохранятся в архиве, исправления перестанут применяться.`)) return;
      try {
        await api(`/api/dictionary/${word.id}`, { method: 'DELETE' });
        if (selectedWord?.id === word.id) editWord(null);
        await loadDictionary(); notify('Слово перемещено в архив');
      } catch (error) { notify(error.message); }
    }, 'button small-button danger'));
    card.append(actions);
    const examples = element('details', undefined, 'examples');
    examples.append(element('summary', `Аудиопримеры: ${word.examples.length}`));
    if (!word.examples.length) examples.append(element('p', 'Нажмите «Изменить / добавить пример», чтобы загрузить запись.'));
    for (const example of word.examples) {
      const clip = element('div', undefined, 'example');
      clip.append(element('small', `${example.filename} · ${example.duration.toFixed(2)} с`), element('p', example.transcript));
      const player = element('audio'); player.controls = true; player.preload = 'none';
      player.src = `/api/dictionary/examples/${example.id}/audio`;
      player.setAttribute('aria-label', `Произношение слова ${word.word}`); clip.append(player); examples.append(clip);
    }
    card.append(examples); list.append(card);
  }
}
$('#dictionary-search').addEventListener('input', renderDictionary);
$('#word-form').addEventListener('submit', async (event) => {
  event.preventDefault(); const submit = $('button[type=submit]', event.currentTarget);
  if (submit.disabled) return;
  submit.disabled = true;
  const input = { word: $('#word-name').value.trim(), category: $('#word-category').value,
    description: $('#word-description').value, aliases: $('#word-aliases').value.split('\n').map(value => value.trim()).filter(Boolean), active: $('#word-active').checked };
  try {
    const word = await api(selectedWord ? `/api/dictionary/${selectedWord.id}` : '/api/dictionary', { method: selectedWord ? 'PUT' : 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(input) });
    editWord(word); status($('#word-status'), 'Слово сохранено. Ниже можно добавить аудиопример.');
    await loadDictionary(); notify('Слово сохранено');
  } catch (error) { status($('#word-status'), error.message, true); }
  finally { submit.disabled = false; }
});
$('#example-form').addEventListener('submit', async (event) => {
  event.preventDefault(); const submit = $('button[type=submit]', event.currentTarget);
  if (submit.disabled || !selectedWord) return;
  const file = $('#example-file').files[0]; if (!file) return;
  if (file.size > 15 * 1024 * 1024) { status($('#example-status'), 'Максимум для аудиопримера — 15 МБ.', true); return; }
  submit.disabled = true; status($('#example-status'), 'Проверяем и сохраняем аудиопример…');
  const wordId = selectedWord.id;
  const data = new FormData(); data.append('audio', file); data.append('transcript', $('#example-transcript').value);
  try {
    const word = await api(`/api/dictionary/${wordId}/examples`, { method: 'POST', body: data });
    if (selectedWord?.id === wordId) { selectedWord = word; $('#example-file').value = ''; }
    status($('#example-status'), 'Пример сохранён. Его можно прослушать в карточке слова справа.');
    await loadDictionary(); notify('Аудиопример добавлен');
  } catch (error) { status($('#example-status'), error.message, true); }
  finally { submit.disabled = false; }
});
loadSpeakers();

const journalStatuses = { queued: 'В очереди', running: 'Обрабатывается', done: 'Готово', failed: 'Ошибка обработки', interrupted: 'Обработка прервана' };
function recordingDate(value) { return new Date(value).toLocaleString('ru-RU'); }
async function loadJournal() {
  const request = ++journalRequest;
  const list = $('#journal-list');
  try {
    const data = await api(`/api/journal?q=${encodeURIComponent($('#journal-search').value.trim())}&offset=${journalOffset}`);
    if (request !== journalRequest) return;
    $('#journal-count').textContent = `Всего: ${data.all_total} · ${(data.total_bytes / 1024 / 1024).toFixed(1)} МБ аудио · найдено: ${data.total}`;
    list.replaceChildren();
    if (!data.entries.length) list.append(element('p', 'Записей нет. Загрузите диалог или расшифровку — они сохранятся автоматически.', 'muted'));
    for (const entry of data.entries) {
      const row = element('article', undefined, 'journal-row');
      row.append(button(entry.filename, () => openRecording(entry.id), 'button secondary journal-open'));
      row.append(element('small', `${recordingDate(entry.created_at)} · ${journalStatuses[entry.status] || entry.status}`));
      if (entry.participants.length) row.append(element('small', entry.participants.map(friendlyName).join(', ')));
      row.append(element('p', entry.preview || entry.error || (entry.operation === 'dialogue' ? 'Диалог по голосам' : 'Расшифровка'), 'muted'));
      list.append(row);
    }
    const pages = $('#journal-pages'); pages.replaceChildren();
    if (journalOffset > 0) pages.append(button('← Назад', () => { journalOffset = Math.max(0, journalOffset - 30); loadJournal(); }));
    if (journalOffset + data.entries.length < data.total) pages.append(button('Ещё →', () => { journalOffset += 30; loadJournal(); }));
  } catch (error) { if (request === journalRequest) list.textContent = error.message; }
}
async function openRecording(id) {
  selectedRecording = id;
  const node = $('#journal-result'); node.textContent = 'Открываем запись…';
  $('#journal-exports').hidden = true; $('#journal-actions').hidden = true;
  try {
    const entry = await api(`/api/journal/${id}`);
    if (selectedRecording !== id) return;
    $('#journal-title').textContent = entry.filename;
    $('#journal-meta').textContent = `${recordingDate(entry.created_at)} · ${journalStatuses[entry.status]} · ${(entry.bytes / 1024 / 1024).toFixed(1)} МБ`;
    const player = $('#journal-player'); player.pause(); player.src = `/api/journal/${id}/audio`; player.hidden = false;
    const actions = $('#journal-actions'); actions.replaceChildren(); actions.hidden = false;
    const audioLink = element('a', 'Скачать аудио', 'button small-button');
    audioLink.href = `/api/journal/${id}/audio?download=true`; audioLink.setAttribute('download', entry.filename); actions.append(audioLink);
    actions.append(button('Сохранить запись JSON', () => download(`journal-${id}.json`, JSON.stringify(entry, null, 2), 'application/json;charset=utf-8')));
    actions.append(button('Удалить запись', async () => {
      if (!window.confirm(`Удалить «${entry.filename}» вместе с аудио и расшифровкой? Это окончательное удаление, не архив.`)) return;
      player.pause(); player.removeAttribute('src'); player.load();
      try {
        await api(`/api/journal/${id}`, { method: 'DELETE' });
        selectedRecording = null; player.hidden = true; actions.hidden = true; $('#journal-exports').hidden = true;
        $('#journal-title').textContent = 'Выберите запись'; $('#journal-meta').textContent = '';
        node.textContent = 'Запись, аудиофайл и результат удалены.';
        journalOffset = 0; await loadJournal(); notify('Запись удалена окончательно');
      } catch (error) { player.src = `/api/journal/${id}/audio`; notify(error.message); }
    }, 'button small-button danger'));
    if (entry.result) {
      if (entry.operation === 'dialogue') renderDialogue(entry.result, 'journal', '#journal-player');
      else renderTranscript(entry.result, 'journal');
    } else {
      node.replaceChildren(element('p', entry.error || 'Обработка ещё не завершена. Аудиозапись уже сохранена. Нажмите «Обновить».', 'muted'));
    }
  } catch (error) { if (selectedRecording === id) node.textContent = error.message; }
}
$('#refresh-journal').addEventListener('click', async () => { await loadJournal(); if (selectedRecording) openRecording(selectedRecording); });
let journalSearchTimer;
$('#journal-search').addEventListener('input', () => {
  clearTimeout(journalSearchTimer); journalSearchTimer = setTimeout(() => { journalOffset = 0; loadJournal(); }, 250);
});
