const audio = document.getElementById("audio");
const loadBtn = document.getElementById("loadBtn");
const playBtn = document.getElementById("playBtn");
const pauseBtn = document.getElementById("pauseBtn");
const songSelect = document.getElementById("songSelect");
const currentSongEl = document.getElementById("currentSong");
const audioTimeEl = document.getElementById("audioTime");
const scoreTimeEl = document.getElementById("scoreTime");
const scoreContainer = document.getElementById("score");

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

let osmd = null;
let tempomap = [];
let scoreBeats = [];
let currentSongId = songSelect?.value || Object.keys(SONGS)[0];

let syncStarted = false;
let cursorSteps = [];
let lastCursorIndex = -1;

async function loadJson(path) {
  const response = await fetch(path);

  if (!response.ok) {
    throw new Error(`Nepodarilo sa načítať súbor: ${path}`);
  }

  return response.json();
}

function getCurrentSong() {
  return SONGS[currentSongId];
}

function updateCurrentSongLabel() {
  const song = getCurrentSong();
  if (currentSongEl && song) {
    currentSongEl.textContent = song.label;
  }
}

function resetSyncState() {
  tempomap = [];
  scoreBeats = [];
  cursorSteps = [];
  lastCursorIndex = -1;
  audioTimeEl.textContent = "0.00";
  scoreTimeEl.textContent = "0.00";
}

function audioToScoreTime(audioTime) {
  if (tempomap.length === 0) return 0;

  const first = tempomap[0];
  const last = tempomap[tempomap.length - 1];

  if (audioTime <= first.audio_time) return first.score_time;
  if (audioTime >= last.audio_time) return last.score_time;

  for (let i = 0; i < tempomap.length - 1; i++) {
    const a = tempomap[i];
    const b = tempomap[i + 1];

    if (audioTime >= a.audio_time && audioTime <= b.audio_time) {
      const audioDiff = b.audio_time - a.audio_time;
      if (audioDiff === 0) return a.score_time;

      const ratio = (audioTime - a.audio_time) / audioDiff;
      return a.score_time + ratio * (b.score_time - a.score_time);
    }
  }

  return last.score_time;
}

async function loadData(song) {
  const [loadedTempomap, loadedScoreBeats] = await Promise.all([
    loadJson(song.tempomap),
    loadJson(song.scoreBeats),
  ]);

  tempomap = loadedTempomap;
  scoreBeats = loadedScoreBeats;
}

async function loadScore(song) {
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

  buildCursorMap();
}

function getCurrentCursorTime() {
  if (!osmd || !osmd.cursor || !osmd.cursor.Iterator) return null;

  const it = osmd.cursor.Iterator;

  if (typeof it.currentTimeStamp?.RealValue === "number") {
    return it.currentTimeStamp.RealValue;
  }

  if (typeof it.CurrentTimeStamp?.RealValue === "number") {
    return it.CurrentTimeStamp.RealValue;
  }

  if (it.CurrentVoiceEntries && it.CurrentVoiceEntries.length > 0) {
    const ts = it.CurrentVoiceEntries[0].Timestamp;
    if (typeof ts?.RealValue === "number") {
      return ts.RealValue;
    }
  }

  return null;
}

function buildCursorMap() {
  cursorSteps = [];
  lastCursorIndex = -1;

  if (!osmd || !osmd.cursor || scoreBeats.length === 0) return;

  osmd.cursor.reset();

  let prevTime = null;
  let safety = 0;

  while (!osmd.cursor.Iterator.EndReached && safety < 200000) {
    const time = getCurrentCursorTime();

    if (time !== null && (prevTime === null || time > prevTime)) {
      cursorSteps.push({
        stepIndex: cursorSteps.length,
        scoreTime: scoreBeats[cursorSteps.length]?.score_time ?? time,
      });
      prevTime = time;
    }

    osmd.cursor.next();
    safety++;
  }

  osmd.cursor.reset();
}

function findNearestCursorIndex(scoreTime) {
  if (cursorSteps.length === 0) return -1;

  let bestIndex = cursorSteps[0].stepIndex;
  let bestDiff = Math.abs(cursorSteps[0].scoreTime - scoreTime);

  for (let i = 1; i < cursorSteps.length; i++) {
    const diff = Math.abs(cursorSteps[i].scoreTime - scoreTime);

    if (diff < bestDiff) {
      bestDiff = diff;
      bestIndex = cursorSteps[i].stepIndex;
    }
  }

  return bestIndex;
}

function moveCursorToIndex(index) {
  if (!osmd || !osmd.cursor) return;
  if (index < 0 || index === lastCursorIndex) return;

  osmd.cursor.reset();

  for (let i = 0; i < index; i++) {
    if (osmd.cursor.Iterator.EndReached) break;
    osmd.cursor.next();
  }

  lastCursorIndex = index;
}

function moveCursorToScoreTime(scoreTime) {
  const index = findNearestCursorIndex(scoreTime);
  moveCursorToIndex(index);
}

function updateSync() {
  const audioTime = audio.currentTime;
  const scoreTime = audioToScoreTime(audioTime);

  audioTimeEl.textContent = audioTime.toFixed(2);
  scoreTimeEl.textContent = scoreTime.toFixed(2);

  if (osmd && tempomap.length > 0 && cursorSteps.length > 0) {
    moveCursorToScoreTime(scoreTime);
  }

  requestAnimationFrame(updateSync);
}

function startSyncLoop() {
  if (syncStarted) return;
  syncStarted = true;
  requestAnimationFrame(updateSync);
}

async function loadSelectedSong() {
  const song = getCurrentSong();
  if (!song) return;

  try {
    audio.pause();
    resetSyncState();

    audio.src = song.audio;
    audio.load();

    await loadData(song);
    await loadScore(song);

    updateCurrentSongLabel();
    startSyncLoop();
  } catch (error) {
    console.error(error);
    alert(`Nepodarilo sa načítať skladbu „${song.label}“. Skontroluj konzolu.`);
  }
}

loadBtn.addEventListener("click", async () => {
  await loadSelectedSong();
});

songSelect.addEventListener("change", async (event) => {
  currentSongId = event.target.value;
  await loadSelectedSong();
});

playBtn.addEventListener("click", () => {
  audio.play();
});

pauseBtn.addEventListener("click", () => {
  audio.pause();
});

updateCurrentSongLabel();
loadSelectedSong();
