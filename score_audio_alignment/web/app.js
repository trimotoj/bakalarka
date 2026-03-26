const audio = document.getElementById("audio");
const loadBtn = document.getElementById("loadBtn");
const playBtn = document.getElementById("playBtn");
const pauseBtn = document.getElementById("pauseBtn");
const audioTimeEl = document.getElementById("audioTime");
const scoreTimeEl = document.getElementById("scoreTime");
const scoreContainer = document.getElementById("score");

let osmd = null;
let tempomap = [];
let scoreBeats = [];

let syncStarted = false;
let cursorSteps = [];
let lastCursorIndex = -1;

// Načíta JSON súbor a vráti jeho obsah
async function loadJson(path) {
  const response = await fetch(path);
  return response.json();
}

// Na základe tempomapy prepočíta čas v audiu na čas v score
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

// Načíta tempomapu a score beaty zo súborov
async function loadData() {
  tempomap = await loadJson("tempomap.json");
  scoreBeats = await loadJson("score_beats.json");
}

// Načíta MusicXML do OSMD a pripraví cursor mapu
async function loadScore() {
  if (!osmd) {
    osmd = new opensheetmusicdisplay.OpenSheetMusicDisplay(scoreContainer, {
      autoResize: true,
      drawTitle: true,
      followCursor: false,
    });
  }

  await osmd.load("chopin.musicxml");
  osmd.render();

  osmd.cursor.show();
  osmd.cursor.reset();

  buildCursorMap();
}

// Zistí aktuálny čas cursoru v score
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

// Prejde celé score a uloží si jednotlivé kroky cursoru
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

// Nájde cursor index, ktorého čas je najbližšie k danému score času
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

// Posunie cursor na konkrétny index v score
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

// Posunie cursor podľa času v score
function moveCursorToScoreTime(scoreTime) {
  const index = findNearestCursorIndex(scoreTime);
  moveCursorToIndex(index);
}

// Priebežne synchronizuje audio čas, score čas a polohu cursoru
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

// Spustí synchronizačný loop iba raz
function startSyncLoop() {
  if (syncStarted) return;
  syncStarted = true;
  requestAnimationFrame(updateSync);
}

// Načíta dáta aj score a potom spustí synchronizáciu
loadBtn.addEventListener("click", async () => {
  await loadData();
  await loadScore();
  startSyncLoop();
});

// Spustí prehrávanie audia
playBtn.addEventListener("click", () => {
  audio.play();
});

// Pozastaví prehrávanie audia
pauseBtn.addEventListener("click", () => {
  audio.pause();
});