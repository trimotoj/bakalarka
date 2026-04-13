const $ = (id) => document.getElementById(id);

const dom = {
  audio: $("audio"),
  loadBtn: $("loadBtn"),
  playBtn: $("playBtn"),
  pauseBtn: $("pauseBtn"),
  songSelect: $("songSelect"),
  currentSong: $("currentSong"),
  audioTime: $("audioTime"),
  scoreTime: $("scoreTime"),
  score: $("score"),
};

const SONGS = {
  "aka-si-mi-krasna": {
    label: "Aká si mi krásna",
    audio: "data/audio/aka-si-mi-krasna.wav",
    score: "data/score/aka-si-mi-krasna.musicxml",
    tempomap: "data/alignment/aka-si-mi-krasna_tempomap.json",
    scoreBeats: "data/alignment/aka-si-mi-krasna_score_beats.json",
  },
  chopin: {
    label: "Chopin",
    audio: "data/audio/chopin.wav",
    score: "data/score/chopin.musicxml",
    tempomap: "data/alignment/chopin_tempomap.json",
    scoreBeats: "data/alignment/chopin_score_beats.json",
  },
};

const SCORE_SCALE = 4;
const MAX_CURSOR_STEPS = 200000;
const EPS = 1e-9;

const state = {
  currentSongId: dom.songSelect?.value || Object.keys(SONGS)[0],
  osmd: null,
  tempomap: [],
  cursorSteps: [],
  lastCursorIndex: -1,
  loopStarted: false,
};

function getSong() {
  return SONGS[state.currentSongId];
}

function setSongLabel() {
  const song = getSong();
  if (song) {
    dom.currentSong.textContent = song.label;
  }
}

function setTimes(audioTime = 0, scoreTime = 0) {
  dom.audioTime.textContent = audioTime.toFixed(2);
  dom.scoreTime.textContent = scoreTime.toFixed(2);
}

function resetAlignmentState() {
  state.tempomap = [];
  state.cursorSteps = [];
  state.lastCursorIndex = -1;
  setTimes();
}

async function fetchJson(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Nepodarilo sa načítať súbor: ${path}`);
  }
  return response.json();
}

function toFiniteNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function normalizeTempomap(data) {
  if (!Array.isArray(data)) {
    return [];
  }

  const points = data
    .map((point) => ({
      audio_time: toFiniteNumber(point?.audio_time),
      score_time: toFiniteNumber(point?.score_time),
    }))
    .filter((point) => point.audio_time !== null && point.score_time !== null)
    .sort((a, b) => a.audio_time - b.audio_time || a.score_time - b.score_time);

  const merged = [];

  for (const point of points) {
    const last = merged[merged.length - 1];

    if (last && Math.abs(last.audio_time - point.audio_time) < EPS) {
      last.score_time = Math.max(last.score_time, point.score_time);
      continue;
    }

    merged.push(point);
  }

  return merged;
}

function getScoreStart(scoreBeats) {
  if (!Array.isArray(scoreBeats)) {
    return 0;
  }

  const values = scoreBeats
    .map((item) => toFiniteNumber(item?.score_time))
    .filter((value) => value !== null)
    .sort((a, b) => a - b);

  return values[0] ?? 0;
}

function findInsertIndex(items, value, getValue) {
  let lo = 0;
  let hi = items.length - 1;

  while (lo <= hi) {
    const mid = Math.floor((lo + hi) / 2);
    const midValue = getValue(items[mid]);

    if (midValue === value) {
      return mid;
    }

    if (midValue < value) {
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }

  return lo;
}

function findNearestIndex(items, value, getValue) {
  if (items.length === 0) {
    return -1;
  }

  const firstValue = getValue(items[0]);
  const lastIndex = items.length - 1;
  const lastValue = getValue(items[lastIndex]);

  if (value <= firstValue) {
    return 0;
  }

  if (value >= lastValue) {
    return lastIndex;
  }

  const right = findInsertIndex(items, value, getValue);
  const left = right - 1;

  return Math.abs(getValue(items[left]) - value) <=
    Math.abs(getValue(items[right]) - value)
    ? left
    : right;
}

function interpolateFromMap(audioTime) {
  const points = state.tempomap;

  if (points.length === 0) {
    return 0;
  }

  const first = points[0];
  const last = points[points.length - 1];

  if (audioTime <= first.audio_time) {
    return first.score_time;
  }

  if (audioTime >= last.audio_time) {
    return last.score_time;
  }

  const rightIndex = findInsertIndex(points, audioTime, (point) => point.audio_time);
  const left = points[rightIndex - 1];
  const right = points[rightIndex];

  const ratio =
    (audioTime - left.audio_time) / (right.audio_time - left.audio_time);

  return left.score_time + ratio * (right.score_time - left.score_time);
}

function getCursorTimestamp() {
  const iterator = state.osmd?.cursor?.Iterator;
  if (!iterator) {
    return null;
  }

  if (typeof iterator.currentTimeStamp?.RealValue === "number") {
    return iterator.currentTimeStamp.RealValue;
  }

  if (typeof iterator.CurrentTimeStamp?.RealValue === "number") {
    return iterator.CurrentTimeStamp.RealValue;
  }

  const entry = iterator.CurrentVoiceEntries?.[0];
  if (typeof entry?.Timestamp?.RealValue === "number") {
    return entry.Timestamp.RealValue;
  }

  return null;
}

function resetCursor() {
  state.osmd?.cursor?.reset();
  state.lastCursorIndex = 0;
}

function buildCursorSteps(scoreStart = 0) {
  state.cursorSteps = [];
  state.lastCursorIndex = -1;

  if (!state.osmd?.cursor) {
    return;
  }

  state.osmd.cursor.reset();

  const rawTimes = [];
  let previousTime = null;
  let stepCount = 0;

  while (!state.osmd.cursor.Iterator.EndReached && stepCount < MAX_CURSOR_STEPS) {
    const time = getCursorTimestamp();

    if (time !== null && (previousTime === null || time > previousTime)) {
      rawTimes.push(time);
      previousTime = time;
    }

    state.osmd.cursor.next();
    stepCount += 1;
  }

  state.osmd.cursor.reset();

  if (rawTimes.length === 0) {
    return;
  }

  const offset = scoreStart - rawTimes[0] * SCORE_SCALE;

  state.cursorSteps = rawTimes.map((time, index) => ({
    index,
    scoreTime: time * SCORE_SCALE + offset,
  }));
}

function moveCursorTo(index) {
  if (!state.osmd?.cursor || index < 0 || index === state.lastCursorIndex) {
    return;
  }

  if (state.lastCursorIndex === -1 || index < state.lastCursorIndex) {
    resetCursor();
  }

  while (
    state.lastCursorIndex < index &&
    !state.osmd.cursor.Iterator.EndReached
  ) {
    state.osmd.cursor.next();
    state.lastCursorIndex += 1;
  }
}

async function ensureOsmd() {
  if (state.osmd) {
    return state.osmd;
  }

  state.osmd = new opensheetmusicdisplay.OpenSheetMusicDisplay(dom.score, {
    autoResize: true,
    drawTitle: true,
    followCursor: false,
  });

  return state.osmd;
}

async function renderScore(scorePath) {
  const osmd = await ensureOsmd();
  await osmd.load(scorePath);
  osmd.render();
  osmd.cursor.show();
  osmd.cursor.reset();
}

function syncUi() {
  const audioTime = dom.audio.currentTime;
  const scoreTime = interpolateFromMap(audioTime);

  setTimes(audioTime, scoreTime);

  if (state.tempomap.length === 0 || state.cursorSteps.length === 0) {
    return;
  }

  const cursorIndex = findNearestIndex(
    state.cursorSteps,
    scoreTime,
    (step) => step.scoreTime,
  );

  moveCursorTo(cursorIndex);
}

function startSyncLoop() {
  if (state.loopStarted) {
    return;
  }

  state.loopStarted = true;

  const tick = () => {
    syncUi();
    requestAnimationFrame(tick);
  };

  requestAnimationFrame(tick);
}

async function loadSelectedSong() {
  const song = getSong();
  if (!song) {
    return;
  }

  try {
    dom.audio.pause();
    resetAlignmentState();

    dom.audio.src = song.audio;
    dom.audio.load();

    const [tempomapData, scoreBeatsData] = await Promise.all([
      fetchJson(song.tempomap),
      fetchJson(song.scoreBeats),
    ]);

    state.tempomap = normalizeTempomap(tempomapData);

    await renderScore(song.score);
    buildCursorSteps(getScoreStart(scoreBeatsData));

    setSongLabel();
    syncUi();
    startSyncLoop();
  } catch (error) {
    console.error(error);
    alert(`Nepodarilo sa načítať skladbu „${song.label}“. Skontroluj konzolu.`);
  }
}

function attachEvents() {
  dom.loadBtn.addEventListener("click", loadSelectedSong);

  dom.songSelect.addEventListener("change", (event) => {
    state.currentSongId = event.target.value;
    setSongLabel();
    loadSelectedSong();
  });

  dom.playBtn.addEventListener("click", () => dom.audio.play());
  dom.pauseBtn.addEventListener("click", () => dom.audio.pause());

  dom.audio.addEventListener("loadedmetadata", syncUi);
  dom.audio.addEventListener("seeking", syncUi);
  dom.audio.addEventListener("seeked", syncUi);
}

function init() {
  setSongLabel();
  attachEvents();
  loadSelectedSong();
}

init();
