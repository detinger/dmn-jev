# Plan nastavka: od gotovog pipelinea do predaje rada

**Rad:** *From typed probabilistic answers to auditable decision tables: distilling a System One model into DMN with risk-controlled automation*
**Ciljni časopis:** Expert Systems with Applications (Elsevier, Q1)
**Stanje na dan:** 7. 10. 2026.
**Polazni dokument:** [jev-dmn-istrazivacki-plan.md](jev-dmn-istrazivacki-plan.md)

## 1. Gdje smo

| Područje | Stanje |
|---|---|
| Podaci | Gotovo. Korpus od 540.782 CFPB pritužbe, uzorak od 17.674, podjela pool / kalibracija / test, manifest s kontrolnim zbrojevima. |
| Shema pitanja | Prva verzija u [configs/cfpb.yaml](configs/cfpb.yaml): 14 činjeničnih i 2 holistička pitanja. **Još nije provjerena na stvarnom Jevu.** |
| Destilacija u DMN, verifikacija, vrata (LTT), izvoz DMN/BPMN | Implementirano i testirano (11 testova, XSD, `feelin` 100 %). |
| Eksperimenti RQ1–RQ4 | Skripte gotove. Pokrenute su samo sa simuliranim ekstraktorom. |
| Rad | Nacrt od 16 stranica u [paper/main.tex](paper/main.tex). Rezultati su placeholderi `\res{}`, a interpretacije su označene s `TODO`. |
| Nedostaje | Stvarni Jev pozivi, LLM osnovica, provjera redoslijeda opcija, interpretacija rezultata, podaci o autorima, repozitorij. |

Glavna blokada je **TypeSafe API ključ**. Faze 2 do 5 ovise o njemu, a faze 1, 6 i 7 mogu teći paralelno.

## 2. Odluke koje trebate donijeti

| # | Odluka | Preporuka | Zašto je važno |
|---|---|---|---|
| O1 | Drugi javni skup podataka | **Da**, EMSCAD (lažni oglasi za posao, ~18k, stvarne oznake prijevare) | Dva zadatka na jednom izvoru recenzenti mogu smatrati slabim dokazom poopćivosti. Pipeline je generički, pa treba samo nova shema i loader (oko tjedan dana). |
| O2 | Ručna anotacija činjenica | **Da**, 200 slučajeva, 2 anotatora | Provjerava točnost Jevovih činjeničnih odgovora (koncepata). Bez toga recenzent može pitati „jesu li ulazi u tablicu uopće točni?”. |
| O3 | LLM osnovica | **Da**, jedan komercijalni i jedan otvoreni model preko `system-one-adapter` | Plan (RQ4) obećava usporedbu s klasičnim LLM-om. Otvoreni model (npr. preko Ollame, već instalirane) daje ponovljivost bez zatvorenog API-ja. |
| O4 | Koautori i uloge (CRediT) | — | Potrebno za naslovnicu i izjavu o doprinosu. |
| O5 | Odnos s TypeSafeom | — | Ako postoji rani pristup ili besplatni krediti, to ide u izjavu o sukobu interesa. Treba provjeriti i uvjete korištenja (smiju li se objaviti sirovi odgovori i usporedbe). |
| O6 | Izvođenje u Camundi | **Da**, kratka demonstracija (Camunda 8 SaaS trial ili Docker na drugom računalu) | Rad tvrdi izvršivost. Sada je ona dokazana validacijom sheme i neovisnim FEEL interpreterom, ali ne i stvarnim pokretanjem procesa. |
| O7 | Ciljni datum predaje | Oko 10 tjedana nakon dobivanja ključa | Određuje redoslijed proširenja u fazi 5. |

## 3. Faze

Trajanja su u radnim tjednima. Tjedan 1 počinje kad stigne ključ. Faze 1, 6 i 7 mogu početi odmah.

### Faza 1: Priprema bez ključa (odmah, ~1 tj.)

- [ ] `git init`, prvi commit, privatni GitHub repozitorij (javni tek pri predaji).
- [ ] Licenca: MIT za kod. Podaci prema licenci CFPB-a i zrcala (Apache-2.0).
- [ ] Zamrznuti verziju sheme pitanja kao `configs/cfpb.yaml` v1 (tag u gitu). Svaka kasnija promjena dobiva novu verziju i bilježi se.
- [ ] Ako se prihvati O1: loader i shema za EMSCAD (`src/jevdmn/data/emscad.py`, `configs/emscad.yaml`), test sa simuliranim ekstraktorom.
- [ ] Ako se prihvati O3: ekstraktor `extractors/adapter.py` preko `system-one-adapter` (isti format odgovora kao Jev). Lokalni test s Ollama modelom.
- [ ] Ako se prihvati O2: protokol anotacije (upute, 200 nasumičnih slučajeva iz poola, ne iz testa) i jednostavan obrazac (CSV ili mala web-stranica).

**Gotovo kad:** repozitorij postoji, shema je označena v1, novi ekstraktori ili skupovi prolaze simulirani test.

### Faza 2: Provjera na stvarnom Jevu (tj. 1)

- [ ] Ključ u `.env`, pokrenuti `python scripts/extract.py --extractor jev --limit 20` (piše u `*_smoke.jsonl`).
- [ ] Usporediti stvarni format odgovora s pretpostavkama u [src/jevdmn/features.py](src/jevdmn/features.py): ključevi vjerojatnosti za score (`"0"`…`"3"`), `confidence` za choice/score, polje `model` = `jev-1.13.0`.
- [ ] Zabilježiti latenciju, broj ulaznih tokena i jesu li se pojavile greške 429/529.
- [ ] Ručno pregledati odgovore za 20 slučajeva i provjeriti razumije li Jev pitanja doslovno onako kako su zamišljena.

**Gotovo kad:** format odgovara kodu, nema grešaka u pozivima, model je zabilježen.

### Faza 3: Pilot i zamrzavanje sheme (tj. 1–2)

Pilot se radi **samo na poolu**, nikad na kalibracijskom ili testnom skupu, kako bi se izbjeglo curenje informacija u testni dio.

- [ ] Pokrenuti Jev na 500 slučajeva iz poola: `python scripts/extract.py --extractor jev --pilot 500` (piše u `*_pilot.jsonl`).
- [ ] Provjeriti raspodjele odgovora: nouli koji su uvijek blizu 0 ili 1, opcije koje se nikad ne biraju, preferencija prve opcije.
- [ ] Ako je potrebno, preformulirati pitanja (najviše jedan krug izmjena) i označiti shemu kao v2.
- [ ] Ako se prihvati O2: anotacija 200 slučajeva (2 anotatora, Cohenov κ, zatim usuglašavanje).
- [ ] Zamrznuti shemu: hash datoteke `configs/cfpb.yaml` upisati u `results/data_manifest.json` prije punog pokretanja.

**Gotovo kad:** shema je zamrznuta, a promjene i razlozi su zapisani (ulaze u rad kao opis postupka).

### Faza 4: Puno pokretanje i analiza (tj. 2–3)

- [ ] `python scripts/extract.py --extractor jev`: 17.674 zahtjeva, trošak ispod 1 $, odgovori se spremaju u `cache/jev`.
- [ ] `python scripts/run_experiments.py --extractor jev`
- [ ] `python scripts/check_dmn_external.py --extractor jev`: XSD i slaganje s `feelin` moraju biti 100 %.
- [ ] Robusnost na redoslijed opcija: `extract.py --subset 1000 --reverse-options`, zatim `robustness.py`.
- [ ] `python scripts/make_paper_assets.py --extractor jev` i ponovni prijevod rada. Crvenih placeholdera više ne smije biti.
- [ ] Provjere smislenosti:
  - stopa kršenja jamstva u provjeri ponovljenim podjelama mora biti ≤ δ (5 %) za svaki ε;
  - krivulja vjernost/složenost ne smije biti degenerirana;
  - destilirana tablica ne smije biti trivijalna (npr. jedno pravilo „uvijek no_relief”).

**Gotovo kad:** svi brojevi u radu dolaze iz `results/jev`, a sve provjere prolaze.

#### Što ako rezultati nisu povoljni

| Ishod | Reakcija |
|---|---|
| Niska vjernost (npr. ispod 80 %) | Analizirati slučajeve u kojima se tablica i Jev ne slažu i pronaći činjenicu koja nedostaje u shemi (to je i nalaz za rad). Usporediti s RuleFitom ili ansamblom kao gornjom granicom vjernosti, kako stoji u izvornom planu. |
| Vrata kod naknade ne automatiziraju ništa pri ε ≤ 5 % | Očekivano zbog šuma u oznakama. Prikazati kao „vrata se prilagođavaju težini zadatka” i staviti naglasak na ε = 10 % i na kontrast sa zadatkom usmjeravanja. |
| Jev je loše kalibriran | Nalaz za RQ2. Jamstvo LTT-a ne ovisi o kalibraciji, a izotonična kalibracija postaje dio metode. |
| Velika osjetljivost na redoslijed opcija | Uvesti prosjek odgovora nad dvama redoslijedima kao dio metode i ponovno pokrenuti. Trošak je i dalje zanemariv. |

### Faza 5: Proširenja za Q1 (tj. 3–6)

Poredano po vrijednosti za recenziju. Ako vrijeme ne dopušta, najprije se odbacuju posljednje stavke.

1. [ ] **Drugi skup podataka (O1):** isti protokol na EMSCAD-u. U rad ulazi kao treći zadatak.
2. [ ] **LLM osnovica (O3):** isti upiti preko adaptera. Usporedba točnosti, kalibracije, latencije i troška. Uključuje i varijantu „LLM kao učitelj” za destilaciju.
3. [ ] **Vremenska podjela:** učenje na 2023., test na 2024. Provjerava pretpostavku izmjenjivosti iz Propozicije 1 i pokazuje koliko pomak u podacima utječe na pokrivenost i rizik.
4. [ ] **Stabilnost tablica:** bootstrap poola (npr. 50 ponavljanja), mjerenje koliko se pravila i korištenih ulaza mijenja. Važno za tvrdnju o mogućnosti revizije.
5. [ ] **Ablacije:**
   - stablo na tipiziranim vrijednostima ili na vjerojatnostima;
   - vrata bez filtra slaganja;
   - veličina kalibracijskog skupa (250 / 500 / 1.000 / 2.000) u odnosu na pokrivenost;
   - osjetljivost na δ.
6. [ ] **Točnost činjenica (O2):** Jev u odnosu na ljudsku anotaciju, po pitanju.
7. [ ] **Pravednost automatizacije:** pokrivenost i rizik vrata po CFPB oznakama „Older American” i „Servicemember”. Povezati s raspravom o EU AI Actu.
8. [ ] **Izvođenje u Camundi (O6):** Python worker (`pyzeebe`) za zadatak `jev-extract`, pokretanje procesa na 100 slučajeva, mjerenje latencije od početka do kraja i snimka zaslona za rad.

**Gotovo kad:** svako prihvaćeno proširenje ima skriptu, rezultat u `results/` i tablicu ili sliku generiranu u `paper/generated/`.

### Faza 6: Pisanje (tj. 5–8, dijelom paralelno)

- [ ] Napisati sve odlomke označene s `TODO` u [paper/main.tex](paper/main.tex):
  - RQ1: koje činjenice tablice koriste;
  - RQ2: smjer kalibracije;
  - RQ3: kontrast između zadataka;
  - Rasprava i Zaključak: brojke iz rezultata.
- [ ] Sažetak s ključnim brojkama (najviše 250 riječi). Highlights do 85 znakova po točki.
- [ ] Ako se odustane od nekog proširenja, ukloniti odgovarajuće tvrdnje iz uvoda i doprinosa.
- [ ] Grafički sažetak (ESWA ga preporučuje): pojednostavljena slika arhitekture s jednom ključnom brojkom.
- [ ] Prilog: puni tekst pitanja, popis 14 parova issue/sub-issue za red „fraud”, primjer DMN XML-a.
- [ ] Provjera stila: ESWA Guide for Authors (opseg, format literature, izjave).
- [ ] Interna recenzija koautora (1 tj.) i jezična lektura.

### Faza 7: Objava materijala i predaja (tj. 9–10)

- [ ] Javni GitHub repozitorij i arhiva na Zenodu s DOI-jem (kod, shema, ID-jevi pritužbi i podjele, DMN/BPMN modeli, sirovi Jev odgovori ako uvjeti korištenja to dopuštaju).
- [ ] Popuniti izjave u radu: dostupnost podataka (URL + DOI), sukob interesa, financiranje, CRediT.
- [ ] Popratno pismo (cover letter): novost u odnosu na Goossens i sur. (2023., ESWA) i Etinger i sur. (2019.).
- [ ] Prijedlog 3–5 recenzenata (DMN, odlučivanje u BPM-u, selektivna klasifikacija), bez sukoba interesa.
- [ ] Predaja kroz Editorial Manager.

## 4. Vremenski pregled

| Tjedan | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| F1 Priprema (može odmah) | ■ | | | | | | | | | |
| F2 Provjera na Jevu | ■ | | | | | | | | | |
| F3 Pilot i zamrzavanje sheme | ■ | ■ | | | | | | | | |
| F4 Puno pokretanje | | ■ | ■ | | | | | | | |
| F5 Proširenja | | | ■ | ■ | ■ | ■ | | | | |
| F6 Pisanje | | | | | ■ | ■ | ■ | ■ | | |
| F7 Objava i predaja | | | | | | | | | ■ | ■ |

Ukupno je oko 10 tjedana od dobivanja ključa, u odnosu na 23 tjedna u izvornom planu. Faze pripreme, destilacije, vrata i integracije su većinom već gotove.

## 5. Rizici (ažurirano)

| Rizik | Vjerojatnost | Ublažavanje |
|---|---|---|
| Ključ kasni | Srednja | Faze 1 i 5.2 teku s LLM-om ili Ollamom preko adaptera, pa je cijeli pipeline spreman za trenutak kad ključ stigne. |
| Promjena verzije modela tijekom rada | Srednja | Model je fiksiran na `jev-1.13.0`, svaki odgovor se sprema s verzijom, a puno pokretanje radi se u jednom danu. |
| Uvjeti korištenja ne dopuštaju objavu odgovora | Niska do srednja | Provjeriti prije faze 7. U najgorem slučaju objavljuju se samo agregati i kod, a rad to navodi. |
| Šumne oznake (naknada je odluka tvrtke, red je izbor potrošača) | Visoka (poznato) | Već je opisano u ograničenjima rada. Drugi skup podataka (O1) ima čvršće oznake. |
| Recenzent: „zatvoren model, nije ponovljivo” | Visoka | Adapter s otvorenim modelom (O3), objavljeni ID-jevi, odgovori i kontrolni zbrojevi, te metoda formulirana za cijelu klasu modela. |
| Recenzent: „samo jedna domena” | Visoka bez O1 | Prihvatiti O1. |
| CFPB narativi više nisu javni | Već se dogodilo | Javna kopija + spajanje po ID-u. Kontrolni zbrojevi i ID-jevi su u manifestu. |

## 6. Prvi koraci kad stigne ključ

```bash
cp .env.example .env    # upisati TYPESAFE_API_KEY
python scripts/extract.py --extractor jev --limit 20
# pregled odgovora → pilot na poolu (faza 3) → zamrzavanje sheme → puno pokretanje:
python scripts/extract.py --extractor jev
python scripts/run_experiments.py --extractor jev
python scripts/check_dmn_external.py --extractor jev
python scripts/make_paper_assets.py --extractor jev
```
