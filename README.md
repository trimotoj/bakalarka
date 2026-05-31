# Tempomapa hudobného záznamu

Tento repozitár obsahuje praktickú časť bakalárskej práce **Tempomapa hudobného záznamu**.
Cieľom projektu je automaticky zarovnať notový zápis skladby so zvukovou nahrávkou tej istej skladby a vytvoriť **tempomapu**, teda mapovanie medzi časom v notovom zápise a časom vo zvukovej nahrávke.

Implementované riešenie používa offline postup:

1. načítanie notového zápisu vo formáte MusicXML,
2. načítanie zvukovej nahrávky vo formáte WAV,
3. extrakciu chroma príznakov z oboch reprezentácií,
4. výpočet matice nákladov,
5. zarovnanie pomocou algoritmu DTW,
6. vytvorenie tempomapy,
7. export výsledkov a webovú vizualizáciu synchronizácie.

Hlavná aplikácia sa nachádza v priečinku:

```bash
score_audio_alignment_baseline
```

---

## Požiadavky

Na spustenie je potrebné mať nainštalované:

* Python 3.10 alebo novší,
* Git,
* moderný webový prehliadač.

Použité Python knižnice:

* `numpy`
* `scipy`
* `matplotlib`
* `librosa`
* `partitura`
* `mido`
* `pandas`

---

## Inštalácia

Najskôr si naklonuj repozitár:

```bash
git clone https://github.com/trimotoj/bakalarka.git
cd bakalarka/score_audio_alignment_baseline
```

Vytvor virtuálne prostredie:

```bash
python -m venv .venv
```

Aktivuj virtuálne prostredie.

Na Linuxe alebo macOS:

```bash
source .venv/bin/activate
```

Na Windows:

```bash
.venv\Scripts\activate
```

Nainštaluj závislosti:

```bash
pip install numpy scipy matplotlib librosa partitura mido pandas
```

Ak je súbor `requirements.txt` upravený tak, že každá závislosť je na samostatnom riadku, môžeš použiť aj:

```bash
pip install -r requirements.txt
```

---

## Štruktúra projektu

```text
score_audio_alignment_baseline/
├── config/
│   └── songs.json              # zoznam skladieb a ciest k vstupom
├── data/
│   ├── audio/                  # zvukové nahrávky
│   └── score/                  # notové zápisy MusicXML
├── evaluation/                 # referenčné body a vyhodnotenie
├── output/                     # výstupy po spustení výpočtu
├── src/                        # zdrojový kód výpočtovej časti
├── web/                        # webová vizualizácia
├── music_synchronization_analysis.ipynb
└── requirements.txt
```

---

## Spustenie výpočtu tempomapy

Výpočtová časť sa spúšťa z priečinka `score_audio_alignment_baseline`.

### Spracovanie jednej skladby z konfigurácie

```bash
python -m src.main --song chopin
```

Identifikátor skladby musí existovať v súbore:

```text
config/songs.json
```

Príklady dostupných identifikátorov:

```text
chopin
aka-si-mi-krasna
aka-si-mi-krasna-transposed
nebudem-dobry
palenocka
palenocka-live
misatango-kyrie
misatango-gloria
```

### Spracovanie všetkých skladieb

```bash
python -m src.main --all
```

Pri dlhších skladbách môže výpočet trvať výrazne dlhšie, pretože základná implementácia používa plnú maticu nákladov a klasické DTW bez obmedzenia vyhľadávacieho priestoru.

### Spracovanie vlastných vstupov

Použiť sa dá aj vlastný notový zápis a vlastná zvuková nahrávka:

```bash
python -m src.main --score data/score/chopin.musicxml --audio data/audio/chopin.wav --piece-name moj-test
```

Výstupy sa uložia do:

```text
output/moj-test/
```

---

## Výstupy

Po úspešnom výpočte vznikne pre každú skladbu samostatný priečinok:

```text
output/<song-id>/
```

Napríklad:

```text
output/chopin/
```

V ňom sa nachádzajú tri hlavné podpriečinky:

```text
exports/
plots/
analysis/
```

### `exports/`

Obsahuje hlavné exportované výsledky:

```text
tempomap.json       # vyhladená tempomapa používaná vo webovej vizualizácii
tempomap_raw.json   # pôvodná tempomapa bez vyhladenia
score_beats.json    # časové body notového zápisu pre webovú vizualizáciu
path.csv            # zarovnávacia cesta
```

### `plots/`

Obsahuje kontrolné grafy:

```text
score_chroma.png
audio_chroma.png
cost_matrix_with_path.png
tempomap.png
tempomap_raw.png
tempomap_raw_vs_smooth.png
aligned_chromas.png
```

Tieto grafy slúžia na kontrolu kvality zarovnania.

### `analysis/`

Obsahuje medzivýsledky vo formáte `.npy` a tabuľku notových udalostí:

```text
audio_times.npy
audio_chroma.npy
score_times.npy
score_chroma.npy
score_chroma_on_audio_time.npy
path_frames.npy
tempomap_raw.npy
tempomap_smooth.npy
score_notes.csv
```

Tieto súbory sú určené najmä na ďalšiu analýzu v notebooku.

---

## Príprava dát pre webovú vizualizáciu

Po výpočte tempomapy treba pripraviť dáta pre webovú aplikáciu.

Pre všetky skladby:

```bash
python -m src.prepare_web_data
```

Pre jednu skladbu:

```bash
python -m src.prepare_web_data --song chopin
```

Tento skript skopíruje potrebné súbory do priečinka:

```text
web/data/
```

Konkrétne pripraví:

```text
web/data/audio/
web/data/score/
web/data/alignment/
```

---

## Spustenie webovej aplikácie

Webová aplikácia sa nespúšťa otvorením súboru `index.html` priamo z disku, pretože prehliadač by mohol zablokovať načítanie lokálnych JSON, WAV a MusicXML súborov.

Z priečinka `score_audio_alignment_baseline` spusti jednoduchý lokálny server:

```bash
python -m http.server 8000
```

Potom otvor v prehliadači:

```text
http://localhost:8000/web/
```

Vo webovej aplikácii je možné vybrať skladbu, načítať ju a prehrávať zvukovú nahrávku synchronizovane s notovým zápisom. Zelený kurzor v notovom zápise sa posúva podľa vypočítanej tempomapy.

---

## Pridanie novej skladby

Ak chceš pridať novú skladbu, postupuj takto:

1. Skopíruj notový zápis do priečinka:

```text
data/score/
```

2. Skopíruj zvukovú nahrávku do priečinka:

```text
data/audio/
```

3. Pridaj skladbu do súboru:

```text
config/songs.json
```

Príklad:

```json
{
  "songs": {
    "moja-skladba": {
      "score": "data/score/moja-skladba.musicxml",
      "audio": "data/audio/moja-skladba.wav"
    }
  }
}
```

4. Spusti výpočet:

```bash
python -m src.main --song moja-skladba
```

5. Priprav dáta pre web:

```bash
python -m src.prepare_web_data --song moja-skladba
```

6. Ak chceš skladbu zobrazovať aj vo webovej aplikácii, pridaj ju do zoznamu `SONGS` v súbore:

```text
web/app.js
```

---

## Vyhodnotenie na referenčných bodoch

Repozitár obsahuje aj skript na vyhodnocovanie presnosti tempomapy na ručne anotovaných referenčných bodoch.

Referenčné body sú dvojice:

```text
score_time, audio_time_ref
```

kde `score_time` je čas v notovom zápise a `audio_time_ref` je ručne určený zodpovedajúci čas vo zvukovej nahrávke.

Vyhodnotenie počíta napríklad:

* priemernú absolútnu chybu,
* medián absolútnej chyby,
* maximálnu absolútnu chybu,
* priemernú podpísanú chybu,
* smerodajnú odchýlku,
* podiel bodov v toleranciách.

---

## Časté problémy

### `ModuleNotFoundError: No module named 'src'`

Príkazy treba spúšťať z priečinka:

```text
score_audio_alignment_baseline
```

Nie z priečinka `src`.

Správne:

```bash
python -m src.main --song chopin
```

Nesprávne:

```bash
cd src
python main.py
```

### Webová aplikácia nenačíta skladbu

Skontroluj, či boli najskôr vytvorené výstupy:

```bash
python -m src.main --song chopin
```

a potom pripravené dáta pre web:

```bash
python -m src.prepare_web_data --song chopin
```

Následne spusti web cez lokálny server:

```bash
python -m http.server 8000
```

a otvor:

```text
http://localhost:8000/web/
```

### Výpočet trvá dlho

Základná implementácia používa plné DTW. Pri dlhších skladbách preto rastie veľkosť matice nákladov aj čas výpočtu. Toto je očakávané správanie, najmä pri dlhších dielach.

### Chyba pri načítaní súborov

Skontroluj, či cesty v `config/songs.json` zodpovedajú skutočným súborom v priečinkoch `data/audio/` a `data/score/`.

---

## Použitý princíp

Systém prevádza notový zápis aj zvukovú nahrávku do chroma reprezentácie. Chroma vektor má 12 zložiek, ktoré zodpovedajú tónovým triedam. Následne sa medzi rámcami oboch reprezentácií vypočíta matica nákladov a pomocou algoritmu Dynamic Time Warping sa nájde optimálna zarovnávacia cesta. Z tejto cesty sa vytvorí tempomapa, ktorá mapuje čas notového zápisu na čas zvukovej nahrávky.

---

## Autor

Timotej Peťko
Bakalárska práca: **Tempomapa hudobného záznamu**
Fakulta matematiky, fyziky a informatiky
Univerzita Komenského v Bratislave
