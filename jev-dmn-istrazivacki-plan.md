# Od tipiziranih probabilističkih odluka do DMN pravila

**Radni naslov:** Destilacija i usmjeravanje prema pouzdanosti u BPMN procesima

## 1. Problem

Modeli poput Jeva donose odluke brzo i jeftino, ali su crne kutije: organizacija ne može revidirati pravilo po kojem je odluka donesena. DMN tablice su transparentne i izvršive, ali ne mogu čitati nestrukturirani ulaz, a ručno pisanje pravila je sporo. Nedostaje metoda koja spaja ta dva svijeta i kaže kada se odluci modela smije vjerovati bez čovjeka.

**Praznina:** postojeći radovi prevode stabla naučena na povijesnim podacima u DMN (Etinger i sur., 2019.; DM45, Leewis i sur., 2024.). Prema dosadašnjem pregledu, nitko ne koristi model s tipiziranim izlazom i vjerojatnostima kao izvor oznaka i kao izvor signala pouzdanosti unutar same DMN logike.

## 2. Istraživačka pitanja

- **RQ1 (destilacija):** Koliko vjerno DMN tablica, naučena iz Jevovih oznaka, reproducira odluke modela, i uz koju složenost tablice?
- **RQ2 (kalibracija):** Jesu li Jevove vjerojatnosti dovoljno kalibrirane da se pragovi mogu koristiti kao pravila?
- **RQ3 (usmjeravanje):** Koliki se udio slučajeva može automatizirati uz zadanu gornju granicu pogreške, ako se slučajevi niske pouzdanosti šalju čovjeku?
- **RQ4 (isplativost):** Kako se hibrid uspoređuje sa samim Jevom, klasičnim LLM-om i tablicom naučenom na stvarnim oznakama, po točnosti, latenciji i trošku?

## 3. Arhitektura

DRD ima tri povezane odluke:

1. **Ekstrakcija (Jev).** Nestrukturirani slučaj ulazi, a izlazi su tipizirani odgovori na razložena pitanja, svaki s vjerojatnošću. U DMN-u je to vanjski model znanja; Jevova shema preslikava se na DMN definicije tipova.
2. **Vrata pouzdanosti (DMN tablica).** Ulazi su vjerojatnosti, izlaz je `automatski` ili `ručna provjera`.
3. **Poslovna odluka (destilirana DMN tablica).** Ulazi su tipizirani odgovori iz koraka 1, izlaz je konačna odluka.

U BPMN-u: servisni zadatak poziva Jev, zadatak poslovnog pravila izvršava DRD, a ekskluzivni prijelaz vodi u automatski nastavak ili u korisnički zadatak. Odluke čovjeka spremaju se kao nove oznake za sljedeću iteraciju.

Važna posljedica: Jev ostaje u izvođenju samo kao ekstraktor obilježja, a sama logika odlučivanja je čitljiva i revizijski provjerljiva.

## 4. Metoda

### A. Destilacija u pravila

1. Definirati shemu: 8–15 razloženih pitanja (boolean, enum, broj) i konačnu odluku. Jev podržava kardinalnost do 255, što ograničava enum izlaze.
2. Jev označi veliki skup slučajeva (10⁴–10⁵).
3. Naučiti stablo odluke na tipiziranim odgovorima, s ciljem jednakim Jevovoj konačnoj odluci. Dubina i minimalna veličina lista su parametri kompromisa vjernost/složenost.
4. Prevesti putanje stabla u DMN tablicu (proširenje skripta iz 2019. na FEEL raspone i enum vrijednosti).
5. Verificirati i pojednostaviti: preklapanja, nepokrivena područja, spajanje pravila (Calvanese i sur., 2018.), uz politiku pogotka Unique.

### B. Pouzdanost kao ulaz

1. Na skupu sa stvarnim oznakama izmjeriti kalibraciju (ECE, dijagrami pouzdanosti). Ako je slaba, primijeniti naknadnu kalibraciju (izotonična regresija).
2. Pragove ne birati ručno. Dvije varijante za usporedbu:
   - **Naučena vrata:** drugo, plitko stablo predviđa hoće li se Jev složiti sa stvarnom oznakom; ulazi su vjerojatnosti po polju. Njegovi rezovi postaju pravila tipa `p_odluka >= 0.87`.
   - **Kontrola rizika:** prag se bira na kalibracijskom skupu tako da pogreška automatiziranih slučajeva ostane ispod zadanog ε (konformna kontrola rizika).
3. Dodati signal neslaganja: ako se Jev i destilirana tablica razlikuju, slučaj ide čovjeku.

## 5. Evaluacija

| Dimenzija | Metrika |
|---|---|
| Vjernost | slaganje tablice s Jevom (%), Cohenov κ |
| Točnost | točnost i F1 prema stvarnim oznakama |
| Kalibracija | ECE, Brierov rezultat |
| Usmjeravanje | krivulja rizik–pokrivenost, stopa automatizacije pri ε = 1 %, 2 %, 5 % |
| Složenost | broj pravila, broj ulaza, prosječna duljina pravila |
| Ispravnost | broj preklapanja i rupa prije i poslije pojednostavljenja |
| Operativno | latencija po slučaju, trošak po 1000 slučajeva, opterećenje čovjeka |

**Osnovice za usporedbu:** sam Jev; LLM sa strukturiranim izlazom; stablo naučeno na stvarnim oznakama; hibrid bez vrata pouzdanosti.

**Podaci:** potreban je tekst uz stvarnu oznaku. Prijedlog: jedan javni skup (npr. pritužbe potrošača s kategorijom ili zahtjevi za podršku s usmjeravanjem) i jedan stvarni proces iz vlastite institucije, anonimiziran. Dva skupa daju odgovor na pitanje poopćivosti.

## 6. Plan implementacije

| Faza | Trajanje | Rezultat |
|---|---|---|
| 1. Priprema | 3 tj. | pristup API-ju, odabir procesa, sheme pitanja, podjela podataka (učenje / kalibracija / test) |
| 2. Označavanje | 2 tj. | Jevovi odgovori i vjerojatnosti za sve slučajeve, zapis troška i latencije |
| 3. Destilacija | 4 tj. | generator stablo → DMN XML, verifikacija i pojednostavljenje, krivulja vjernost/složenost |
| 4. Vrata pouzdanosti | 3 tj. | analiza kalibracije, obje varijante pragova, DMN tablica vrata |
| 5. Integracija | 3 tj. | BPMN proces u Camundi s DRD-om, korisničkim zadatkom i povratnom petljom |
| 6. Eksperimenti | 4 tj. | sve osnovice, ablacije, statistički testovi |
| 7. Pisanje | 4 tj. | rukopis i javni repozitorij |

Ukupno: oko 23 tjedna.

**Alati:** Python, scikit-learn, Camunda (Modeler i izvršni sustav), dmn-js za verifikaciju, TypeSafe API.

## 7. Rizici

- **Rani pristup i zatvoren model.** Ponovljivost je slaba točka. Ublažavanje: TypeSafe ima otvoreni adapter (`system-one-adapter-python`) koji ograničava LLM-ove na strukturirane odluke kompatibilne s njihovim API-jem, pa se cijeli cjevovod može ponoviti s otvorenim modelom. Rad treba govoriti o klasi modela, a Jev je jedna instanca.
- **Loša kalibracija.** Ako RQ2 ispadne negativno, to je i dalje objavljiv nalaz, a naknadna kalibracija postaje dio metode.
- **Stablo ne hvata Jevovu logiku.** Niska vjernost pri razumnoj veličini tablice; tada usporediti s ekstrakcijom pravila iz ansambla.
- **Curenje oznaka.** Pragovi se moraju birati na skupu odvojenom od testnog.
- **Promjena verzije modela** tijekom istraživanja: bilježiti verziju i datum svakog poziva te spremiti sve sirove odgovore.

## 8. Očekivani doprinos

1. Metoda destilacije tipiziranih probabilističkih odluka u verificirane DMN tablice.
2. Obrazac "vrata pouzdanosti" u DMN-u s pragovima izvedenima iz podataka i jamstvom na stopu pogreške.
3. Prva neovisna procjena kalibracije modela ove klase u poslovnom procesu.
4. Otvoreni alat i referentni BPMN/DMN modeli.

## Literatura i izvori

- Etinger, D., Simić, S. D., Buljubašić, L. (2019). Automated Decision-Making with DMN: from Decision Trees to Decision Tables. MIPRO 2019. https://www.bib.irb.hr/995284
- Leewis, S., Smit, K., Versendaal, J. (2024). Discovering operational decisions from data. *Decision* 51(4), 417–436. https://link.springer.com/article/10.1007/s40622-024-00402-2
- Calvanese, D. i sur. (2018). Semantics, Analysis and Simplification of DMN Decision Tables. *Information Systems* 78, 112–125. https://www.inf.unibz.it/~calvanese/papers-html/IS-2018.html
- Holistic discovery of decision models from process execution data (P-MInD). *Knowledge-Based Systems* 301. https://datalearner.com/academic/journal-papers/0950-7051/volumes-and-issues/301/paper-detail/8494
- Bork, D., Ali, S. J., Dinev, G. M. (2023). AI-Enhanced Hybrid Decision Management. *BISE* 65(2), 179–199. https://aisel.aisnet.org/bise/vol65/iss2/6
- Almeida, D. (2026). Introducing System One Models & Jev. TypeSafe AI. https://typesafe.ai/blog/introducing-system-one-models-and-jev
