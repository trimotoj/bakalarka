const $ = (id) => document.getElementById(id);

const audio = $("audio");
const loadBtn = $("loadBtn");
const playBtn = $("playBtn");
const pauseBtn = $("pauseBtn");
const songSelect = $("songSelect");
const currentSongEl = $("currentSong");
const audioTimeEl = $("audioTime");
const scoreTimeEl = $("scoreTime");
const scoreContainer = $("score");

const SONGS = {
  "aka-si-mi-krasna": {
    label: "Aká si mi krásna",
    audio: "data/audio/aka-si-mi-krasna.wav",
    score: "data/score/aka-si-mi-krasna.musicxml",
    tempomap: "data/alignment/aka-si-mi-krasna_tempomap_bidirectional.json",
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

let osmd = null;
let currentSongId = songSelect?.value || Object.keys(SONGS)[0];
let tempomap = [];
let cursorSteps = [];
let lastCursorIndex = -1;
let loopStarted = false;

function getSong() {
  return SONGS[currentSongId];
}

function setSongLabel() {
  const song = getSong();
  if (song && currentSongEl) currentSongEl.textContent = song.label;
}

async function loadJson(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Nepodarilo sa načítať súbor: ${path}`);
  }
  return response.json();
}

function resetState() {
  tempomap = [];
  cursorSteps = [];
  lastCursorIndex = -1;

  if (audioTimeEl) audioTimeEl.textContent = "0.00";
  if (scoreTimeEl) scoreTimeEl.textContent = "0.00";
}

function normalizeTempomap(data) {
  if (!Array.isArray(data)) return [];

  const points = data
    .filter(
      (p) =>
        p &&
        Number.isFinite(Number(p.audio_time)) &&
        Number.isFinite(Number(p.score_time)),
    )
    .map((p) => ({
      audio_time: Number(p.audio_time),
      score_time: Number(p.score_time),
    }))
    .sort((a, b) => a.audio_time - b.audio_time || a.score_time - b.score_time);

  const result = [];
  const eps = 1e-9;

  for (const point of points) {
    const last = result[result.length - 1];

    if (last && Math.abs(last.audio_time - point.audio_time) < eps) {
      last.score_time = Math.max(last.score_time, point.score_time);
    } else {
      result.push(point);
    }
  }

  return result;
}

function getFirstScoreTime(scoreBeats) {
  if (!Array.isArray(scoreBeats)) return 0;

  const valid = scoreBeats
    .map((p) => Number(p?.score_time))
    .filter(Number.isFinite)
    .sort((a, b) => a - b);

  return valid[0] ?? 0;
}

function audioToScoreTime(audioTime) {
  if (tempomap.length === 0) return 0;

  const first = tempomap[0];
  const last = tempomap[tempomap.length - 1];

  if (audioTime <= first.audio_time) return first.score_time;
  if (audioTime >= last.audio_time) return last.score_time;

  let lo = 0;
  let hi = tempomap.length - 1;

  while (lo <= hi) {
    const mid = Math.floor((lo + hi) / 2);
    const t = tempomap[mid].audio_time;

    if (t === audioTime) return tempomap[mid].score_time;
    if (t < audioTime) lo = mid + 1;
    else hi = mid - 1;
  }

  const left = tempomap[hi];
  const right = tempomap[lo];
  const ratio = (audioTime - left.audio_time) / (right.audio_time - left.audio_time);

  return left.score_time + ratio * (right.score_time - left.score_time);
}

function getCursorTime() {
  const it = osmd?.cursor?.Iterator;
  if (!it) return null;

  if (typeof it.currentTimeStamp?.RealValue === "number") {
    return it.currentTimeStamp.RealValue;
  }

  if (typeof it.CurrentTimeStamp?.RealValue === "number") {
    return it.CurrentTimeStamp.RealValue;
  }

  const entry = it.CurrentVoiceEntries?.[0];
  if (typeof entry?.Timestamp?.RealValue === "number") {
    return entry.Timestamp.RealValue;
  }

  return null;
}

function buildCursorMap(scoreStart = 0) {
  cursorSteps = [];
  lastCursorIndex = -1;

  if (!osmd?.cursor) return;

  osmd.cursor.reset();

  const rawTimes = [];
  let prev = null;
  let steps = 0;

  while (!osmd.cursor.Iterator.EndReached && steps < MAX_CURSOR_STEPS) {
    const time = getCursorTime();

    if (time !== null && (prev === null || time > prev)) {
      rawTimes.push(time);
      prev = time;
    }

    osmd.cursor.next();
    steps++;
  }

  osmd.cursor.reset();

  if (rawTimes.length === 0) return;

  const offset = scoreStart - rawTimes[0] * SCORE_SCALE;

  cursorSteps = rawTimes.map((time, index) => ({
    index,
    scoreTime: time * SCORE_SCALE + offset,
  }));
}

function findNearestCursorIndex(scoreTime) {
  if (cursorSteps.length === 0) return -1;

  let lo = 0;
  let hi = cursorSteps.length - 1;

  if (scoreTime <= cursorSteps[0].scoreTime) return 0;
  if (scoreTime >= cursorSteps[hi].scoreTime) return hi;

  while (lo <= hi) {
    const mid = Math.floor((lo + hi) / 2);
    const t = cursorSteps[mid].scoreTime;

    if (t === scoreTime) return mid;
    if (t < scoreTime) lo = mid + 1;
    else hi = mid - 1;
  }

  const left = hi;
  const right = lo;

  return Math.abs(cursorSteps[left].scoreTime - scoreTime) <=
    Math.abs(cursorSteps[right].scoreTime - scoreTime)
    ? left
    : right;
}

function moveCursorTo(index) {
  if (!osmd?.cursor || index < 0 || index === lastCursorIndex) return;

  if (lastCursorIndex === -1 || index < lastCursorIndex) {
    osmd.cursor.reset();
    lastCursorIndex = 0;
  }

  while (lastCursorIndex < index && !osmd.cursor.Iterator.EndReached) {
    osmd.cursor.next();
    lastCursorIndex++;
  }
}

function syncNow() {
  const audioTime = audio.currentTime;
  const scoreTime = audioToScoreTime(audioTime);

  if (audioTimeEl) audioTimeEl.textContent = audioTime.toFixed(2);
  if (scoreTimeEl) scoreTimeEl.textContent = scoreTime.toFixed(2);

  if (tempomap.length > 0 && cursorSteps.length > 0) {
    moveCursorTo(findNearestCursorIndex(scoreTime));
  }
}

function startLoop() {
  if (loopStarted) return;
  loopStarted = true;

  const tick = () => {
    syncNow();
    requestAnimationFrame(tick);
  };

  requestAnimationFrame(tick);
}

async function loadSelectedSong() {
  const song = getSong();
  if (!song) return;

  try {
    audio.pause();
    resetState();

    audio.src = song.audio;
    audio.load();

    const [rawTempomap, rawScoreBeats] = await Promise.all([
      loadJson(song.tempomap),
      loadJson(song.scoreBeats),
    ]);

    tempomap = normalizeTempomap(rawTempomap);
    const scoreStart = getFirstScoreTime(rawScoreBeats);

    if (!osmd) {
      osmd = new opensheetmusicdisplay.OpenSheetMusicDisplay(scoreContainer, {
        autoResize: true,
        drawTitle: true,
        followCursor: false,
      });
    }

    await osmd.load(song.score);
    osmd.render();
    osmd.cursor.show();
    osmd.cursor.reset();

    buildCursorMap(scoreStart);
    setSongLabel();
    syncNow();
    startLoop();
  } catch (error) {
    console.error(error);
    alert(`Nepodarilo sa načítať skladbu „${song.label}“. Skontroluj konzolu.`);
  }
}

loadBtn.addEventListener("click", loadSelectedSong);

songSelect.addEventListener("change", (event) => {
  currentSongId = event.target.value;
  loadSelectedSong();
});

playBtn.addEventListener("click", () => audio.play());
pauseBtn.addEventListener("click", () => audio.pause());

audio.addEventListener("seeking", syncNow);
audio.addEventListener("seeked", syncNow);
audio.addEventListener("loadedmetadata", syncNow);

setSongLabel();
loadSelectedSong();