# Run log

Recorded console and table output of the executed notebook, reproduced here so the
evidence behind each reported number is readable without a notebook viewer. The run
is the one reported in the report: Python 3.11.9, random seed 8851,
261,467 documents after cleaning.

Tabular output is abbreviated to its text form; the rendered tables and all sixteen
figures are in `notebooks/consumer_interest_pipeline.ipynb` and `docs/executed_notebook.html`.


## 1. Environment and Parameters

```text
python               : 3.11.9
pandas                : 3.0.5
numpy                 : 2.4.6
scikit-learn          : 1.9.0
bertopic              : 0.17.4
umap-learn            : 0.5.12
hdbscan               : 0.8.44
sentence-transformers : 6.0.0
pytrends              : 4.9.2
random seed          : 8851
```


## 2. Data Preparation

```text
Records            : 312,791
Calendar range     : 2021-07-25 to 2021-08-08 (15 days)
Dates collected    : 9
Dates not collected: 6
```

```text
| Date | Documents | Hours Collected | Share (%) | Status |
| --- | --- | --- | --- | --- |
| 2021-07-25 | 500 | 3 | 0.2% | Collected |
| 2021-07-26 | 47,000 | 12 | 15.0% | Collected |
| 2021-07-27 | 62,499 | 16 | 20.0% | Collected |
| 2021-07-28 | 32,499 | 7 | 10.4% | Collected |
| 2021-07-29 | 30,000 | 8 | 9.6% | Collected |
| 2021-07-30 | 32,497 | 9 | 10.4% | Collected |
| 2021-07-31 | 51,415 | 14 | 16.4% | Collected |
| 2021-08-01 | 33,885 | 9 | 10.8% | Collected |
| 2021-08-02 | — | — | — | Not Collected |
| 2021-08-03 | — | — | — | Not Collected |
| 2021-08-04 | — | — | — | Not Collected |
| 2021-08-05 | — | — | — | Not Collected |
| 2021-08-06 | — | — | — | Not Collected |
| 2021-08-07 | — | — | — | Not Collected |
| 2021-08-08 | 22,496 | 6 | 7.2% | Collected |
```


## 3. Corpus Cleaning

```text
| Operation | Retained | Removed | % of Initial | Rationale |
| --- | --- | --- | --- | --- |
| Timestamped records | 312,791 | — | 100.0% | Records retained after timestamp parsing |
| Duplicate identifiers removed | 307,199 | 5,592 | 98.2% | Overlapping collection runs introduce repeated records |
| Empty text removed | 307,199 | 0 | 98.2% | Records without content cannot be assigned to a topic |
| Retweets removed | 307,199 | 0 | 98.2% | Amplification is not independent discussion |
| Text normalised | 307,199 | 0 | 98.2% | Applied before deduplication so near-duplicates are detected |
| Duplicate text removed | 267,853 | 39,346 | 85.6% | Records identical once punctuation and URLs are stripped |
| Records under 4 tokens removed | 261,467 | 6,386 | 83.6% | Insufficient tokens for reliable cluster assignment |
```

```text
Analysis corpus: 261,467 documents (83.6% of timestamped records) from 175,488 accounts
```


## 4. Unrestricted Topic Discovery (Baseline 0)

```text
Loading weights:   0%|          | 0/103 [00:00<?, ?it/s]
```

```text
Corpus            : 261,467 documents
Topics identified : 291
Outlier documents : 39.5%
```

```text
| Topic | Documents | % of Corpus | Representative Terms |
| --- | --- | --- | --- |
| 0 | 9,873 | 3.78% | tokyo, japan, tokyoolympics, japanese, day, games, olympicgames, medals |
| 1 | 7,585 | 2.90% | she, her, girl, was, that, did, but, is |
| 2 | 3,433 | 1.31% | he, him, his, guy, was, man, at, be |
| 3 | 3,038 | 1.16% | biles, simone, mental, health, withdraws, around, gymnastics, competition |
| 4 | 2,863 | 1.09% | nbc, coverage, app, channel, tv, cable, channels, streaming |
| 5 | 2,624 | 1.00% | watching, watched, watch, haven, love, not, ve, any |
| 6 | 2,555 | 0.98% | gold, medals, silver, medal, bronze, congratulations, winning, won |
| 7 | 2,215 | 0.85% | china, taiwan, chinese, hong, kong, beijing, taipei, haughey |
| 8 | 1,985 | 0.76% | skateboarding, skating, skateboard, skateboarders, skate, skateboarder, skaters, street |
| 9 | 1,870 | 0.72% | gymnastics, gymnasts, gymnast, rhythmic, floor, men, women, artistic |
| 10 | 1,817 | 0.69% | swimming, swim, swimmers, pool, swimmer, synchronized, artistic, water |
| 11 | 1,767 | 0.68% | volleyball, beach, italy, vs, china, turkey, match, brazil |
| 12 | 1,679 | 0.64% | kids, dad, old, age, mom, olds, parents, my |
| 13 | 1,624 | 0.62% | song, songs, music, played, blackpink, kpop, playing, theme |
| 14 | 1,620 | 0.62% | hidilyn, diaz, philippines, filipino, hidilyndiaz, first, gold, ever |
| 15 | 1,618 | 0.62% | archery, archer, korea, archers, atanu, das, korean, san |
| 16 | 1,608 | 0.61% | amp, field, track, up, you, on, we, to |
| 17 | 1,592 | 0.61% | women, female, gender, woman, ladies, men, biological, male |
| 18 | 1,583 | 0.61% | djokovic, novak, slam, nadal, federer, tennis, golden, he |
| 19 | 1,496 | 0.57% | india, indian, indians, medals, proud, performance, medal, contingent |
| 20 | 1,437 | 0.55% | started, happening, this, it, next, wait, olympics, are |
| 21 | 1,424 | 0.54% | hockey, india, indian, ireland, beat, goal, quarter, team |
| 22 | 1,410 | 0.54% | horse, horses, equestrian, dressage, dancing, rider, dance, pommel |
| 23 | 1,336 | 0.51% | you, say, read, this, what, know, your, it |
```


## 5. Domain Term Derivation

```text
Seed set: 4,116 documents from 3 seed terms
```

```text
| Term | Frequency | % of Seed Set | Term | Frequency | % of Seed Set | Term | Frequency | % of Seed Set |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gymnastics | 2,932 | 71.2% | women | 278 | 6.8% | time | 129 | 3.1% |
| olympics | 2,502 | 60.8% | watching | 265 | 6.4% | more | 127 | 3.1% |
| biles | 958 | 23.3% | watch | 258 | 6.3% | men's | 121 | 2.9% |
| simone | 936 | 22.7% | withdraws | 245 | 6.0% | simonebiles | 118 | 2.9% |
| around | 829 | 20.1% | with | 221 | 5.4% | compete | 115 | 2.8% |
| gymnast | 808 | 19.6% | have | 216 | 5.2% | live | 113 | 2.7% |
| tokyo | 716 | 17.4% | after | 209 | 5.1% | only | 111 | 2.7% |
| team | 639 | 15.5% | mental | 199 | 4.8% | when | 110 | 2.7% |
| from | 626 | 15.2% | just | 197 | 4.8% | routine | 107 | 2.6% |
| final | 532 | 12.9% | will | 197 | 4.8% | sports | 105 | 2.6% |
| olympic | 370 | 9.0% | about | 179 | 4.3% | love | 105 | 2.6% |
| this | 320 | 7.8% | what | 151 | 3.7% | rhythmic | 104 | 2.5% |
| that | 306 | 7.4% | they | 149 | 3.6% | event | 102 | 2.5% |
| individual | 305 | 7.4% | finals | 145 | 3.5% | first | 100 | 2.4% |
| gold | 304 | 7.4% | like | 144 | 3.5% | health | 99 | 2.4% |
| women's | 299 | 7.3% | swimming | 140 | 3.4% | sport | 98 | 2.4% |
| competition | 289 | 7.0% | medal | 137 | 3.3% |  |  |  |
```


## 6. Domain Filtering

```text
Unambiguous terms : 23
Ambiguous terms   : 6
Consumer vocabulary in filter: none
```


## 6.1 Per-term coverage

```text
| Term | Documents Matched | Term | Documents Matched | Term | Documents Matched |
| --- | --- | --- | --- | --- | --- |
| simone biles | 3,151 | ring | 146 | dismount | 18 |
| gymnastics | 2,873 | rhythmic gymnastics | 93 | high bar | 17 |
| all around | 810 | uneven bars | 84 | parallel bars | 16 |
| gymnast | 800 | floor routine | 74 | balance beam | 16 |
| gymnasts | 444 | artistic gymnastics | 67 | xiao ruoteng | 14 |
| vault | 352 | beam | 64 | rebeca andrade | 10 |
| floor | 274 | pommel | 47 | nikita nagornyy | 9 |
| bars | 170 | hashimoto | 34 | carlos yulo | 7 |
| sunisa lee | 168 | floor exercise | 21 | still rings | 2 |
| rings | 151 | tumbling | 19 |  |  |
```

```text
Terms matching no documents: none
```


## 6.2 Resulting discourse set

```text
Gymnastics discourse set: 7,129 documents (2.73% of the cleaned corpus) from 5,982 accounts
```

```text
| Date | Documents | % of Set | Status |
| --- | --- | --- | --- |
| 2021-07-25 | 2 | 0.0% | Collected |
| 2021-07-26 | 836 | 11.7% | Collected |
| 2021-07-27 | 1,541 | 21.6% | Collected |
| 2021-07-28 | 1,988 | 27.9% | Collected |
| 2021-07-29 | 529 | 7.4% | Collected |
| 2021-07-30 | 554 | 7.8% | Collected |
| 2021-07-31 | 860 | 12.1% | Collected |
| 2021-08-01 | 611 | 8.6% | Collected |
| 2021-08-02 | — | — | Not Collected |
| 2021-08-03 | — | — | Not Collected |
| 2021-08-04 | — | — | Not Collected |
| 2021-08-05 | — | — | Not Collected |
| 2021-08-06 | — | — | Not Collected |
| 2021-08-07 | — | — | Not Collected |
| 2021-08-08 | 208 | 2.9% | Collected |
```


## 6.3 Filter validation

```text
Accepted on ambiguous term with context: 45
   + @CoachNixonTF Gold Medal Lock-Mondo Duplantis Pole Vault Shock The World Pick - Randolph
   + Suni Lee’s bars routine is soooo beautiful 🤩 #Olympics #Tokyo2020
   + @mykaylaskinner Gurrrrl that Floor and Beam routine was the best I’ve ever seen you do…y
   + Gelya really waited all year until the olympics to hit a beam routine. Queen of causing 
   + The main thing i need to happen at these olympics is a floor medal for vanessa ferrari

Rejected, ambiguous term without context: 625
   - #buy  Egnaro Silicone Wedding Ring for Men, Breathable #Egnaro      https://t.co/dffis3P
   - When any team gets on the floor against Team USA this Olympics https://t.co/3OzV8dmw9v
   - Jrue Holiday went from celebrating his championship ring to being connected with Team US
   - After a courtroom win and a loss in the ring, boxer Mandy Bujold's Olympic dream closes 
   - The basketball 🏀 in the Olympics has an off sound as it bounces. It’s like a watermelon
```


## 7. Model Selection

```text
min_topic_size=  15  topics= 73  outliers=37.6%  largest= 3.7%
min_topic_size=  25  topics= 48  outliers=37.7%  largest= 6.7%
min_topic_size=  50  topics=  9  outliers= 5.0%  largest=43.6%
min_topic_size= 100  topics=  5  outliers= 3.8%  largest=43.6%
```


## 7.1 Comparative evaluation

```text
A seed 8851: 48 topics, 37.7% outliers
A seed 7: 37 topics, 22.8% outliers
A seed 2021: 49 topics, 40.1% outliers
B seed 8851: 73 topics, 37.6% outliers
B seed 7: 69 topics, 34.6% outliers
B seed 2021: 66 topics, 35.7% outliers
NMF seed 8851: 48 topics
NMF seed 7: 48 topics
NMF seed 2021: 48 topics
```

```text
| Model | Setting | Topics | Coherence | Diversity | Seed Consistency | Outlier Share |
| --- | --- | --- | --- | --- | --- | --- |
| BERTopic A | min\_topic\_size=25 | 48 | -2.273000 | 0.823000 | 0.433000 | 37.7% |
| BERTopic B | min\_topic\_size=15 | 73 | -2.097000 | 0.844000 | 0.633000 | 37.6% |
| NMF | n\_topics=48 | 48 | -3.823000 | 0.820000 | 0.741000 | n/a |
```


## 8. Topic Discovery

```text
Documents         : 7,129
Topics identified : 73
Outlier documents : 2,680 (37.6%)
min_topic_size    : 15 (0.21% of the set)
```

```text
| Topic | Documents | % of Set | Representative Terms |
| --- | --- | --- | --- |
| 0 | 263 | 3.7% | lee, sunisa, suni, hmong, gold, community, wins, medal |
| 1 | 253 | 3.5% | accomplishments, biles, simone, excels, pulled, team, comforted, hern |
| 2 | 251 | 3.5% | diving, swimming, love, tennis, skating, favorite, track, favourite |
| 3 | 245 | 3.4% | simone, biles, columnist, resisting, power, decision, rises, relationship |
| 4 | 243 | 3.4% | simone, biles, people, care, mad, criticizing, dropping, drop |
| 5 | 210 | 2.9% | mccallum, girl, grace, routine, niece, greatest, hernandez, laurie |
| 6 | 158 | 2.2% | withdraws, withdrawn, competition, individual, star, superstar, biles, final |
| 7 | 157 | 2.2% | health, mental, focus, importance, excuse, issues, citing, prioritizing |
| 8 | 101 | 1.4% | simonebiles, hope, okay, willingly, simone, forever, goldmedal, understand |
| 9 | 95 | 1.3% | costa, rican, luciana, alvarado, lives, tribute, matter, black |
| 10 | 81 | 1.1% | health, mental, focus, concerns, pulls, prioritising, praised, wellness |
| 11 | 80 | 1.1% | oksana, chusovitina, ovation, standing, eighth, old, uzbekistan, year |
| 12 | 73 | 1.0% | tokyoolympics, artistic, inflection, captivating, societal, layden, moments, marks |
| 13 | 71 | 1.0% | withdraws, individual, competition, simone, spicy, biles, additional, final |
| 14 | 67 | 0.9% | health, mental, intelligence, lesson, emotional, decision, protect, message |
| 15 | 67 | 0.9% | arms, hugging, hairdryer, best, scary, amazed, unreal, man |
| 16 | 66 | 0.9% | rhythmic, rhythmicgymnastics, rhythm, group, peak, intense, ribbon, sport |
| 17 | 63 | 0.9% | mental, category, ang, worthy, na, gold, witness, mentalhealth |
| 18 | 60 | 0.8% | career, gt, couldve, years, kept, maybe, old, parents |
| 19 | 59 | 0.8% | compete, participate, thursday, final, confirms, breaking, simeone, thismorning |
| 20 | 55 | 0.8% | commentary, commentator, commentators, sound, sounds, announcers, commentating, accent |
| 21 | 54 | 0.8% | drops, individual, competition, withdraws, breaking, biles, simone, withdrew |
| 22 | 53 | 0.7% | sexualization, unitards, wears, tired, comfort, german, promote, stand |
| 23 | 53 | 0.7% | women, mat, lock, france, place, nd, takes, gold |
| 24 | 53 | 0.7% | german, unitards, germany, wore, length, norwegian, significance, statement |
| 25 | 51 | 0.7% | russia, russian, nagornyy, nikita, committee, russians, roc, japan |
| 26 | 48 | 0.7% | alarm, wake, woke, sleep, setting, asleep, awake, bed |
| 27 | 46 | 0.6% | suits, body, german, wears, opted, opt, bbc, promote |
| 28 | 45 | 0.6% | pulled, finals, omg, blow, team, simone, biles, reportedly |
| 29 | 44 | 0.6% | understand, lmao, difficulty, safe, learning, cares, know, tf |
| 30 | 44 | 0.6% | hashimoto, daiki, ruoteng, xiao, blind, details, responds, japanese |
| 31 | 43 | 0.6% | grown, cnet, nbc, mat, way, sports, schedule, simone |
| 32 | 43 | 0.6% | skinner, mykayla, heartbroken, run, ending, replace, leave, vault |
| 33 | 43 | 0.6% | pommel, horse, fraser, rings, mt, malone, gbr, joe |
| 34 | 42 | 0.6% | country, rule, federation, sports, international, normal, allowed, sport |
| 35 | 42 | 0.6% | good, skateboarding, bulgarian, tourischeva, dre, blanc, grand, beachvolleyball |
| 36 | 41 | 0.6% | uneven, bars, vault, withdraws, finals, fails, status, attempted |
| 37 | 40 | 0.6% | padukone, deepika, priyanka, chopra, withdraw, focus, laud, bhatt |
| 38 | 39 | 0.5% | twisties, impacting, answered, questions, says, suffering, instagram, experiencing |
| 39 | 39 | 0.5% | music, routines, floor, men, don, male, dance, exercise |
| 40 | 39 | 0.5% | peacock, stream, streaming, nbc, channel, site, begins, et |
| 41 | 38 | 0.5% | parallel, bars, uneven, bar, dismount, jordan, high, chiles |
| 42 | 36 | 0.5% | adhd, meds, medication, banned, ritalin, allowed, drugs, common |
| 43 | 36 | 0.5% | goat, proves, rough, emoji, player, jumped, hubris, pending |
| 44 | 36 | 0.5% | twisties, fear, affects, safety, physical, spin, yips, phenomenon |
| 45 | 35 | 0.5% | djokovic, che, novak, jokes, michael, meltdown, hypocrisy, backlash |
| 46 | 35 | 0.5% | twisties, explains, depths, shuts, led, battling, destroying, reveals |
| 47 | 32 | 0.4% | updates, live, iran, ryan, murphy, routs, lilly, silvers |
| 48 | 32 | 0.4% | vault, planned, stalled, twisting, attempt, perform, leaving, mind |
| 49 | 28 | 0.4% | piers, morgan, nasty, mocked, criticised, host, comments, quitting |
| 50 | 28 | 0.4% | trampoline, trampolinegymnastics, knew, mobot, slaloms, skiing, solved, unlocked |
| 51 | 28 | 0.4% | simonebiles, withdrawn, breaking, defending, competition, medical, tokyoolympics, withdrew |
| 52 | 28 | 0.4% | schedule, blog, july, day, softball, busy, ariake, centre |
| 53 | 26 | 0.4% | defend, title, superstar, aro, withdrew, thursday, american, breaking |
| 54 | 26 | 0.4% | died, paralysed, inches, vaulting, low, neck, paralyzed, quit |
| 55 | 26 | 0.4% | katie, ledecky, caeleb, dressel, sleeping, lost, grabs, updates |
| 56 | 26 | 0.4% | fifth, finish, settles, finishes, missteps, team, men, qualifying |
| 57 | 25 | 0.4% | withdrawal, gmb, avoiding, liz, cambage, alex, danger, defends |
| 58 | 25 | 0.4% | russian, achilles, russians, surgery, months, tore, torn, russia |
| 59 | 24 | 0.3% | carey, jade, oregon, arizona, state, home, future, qualifies |
| 60 | 24 | 0.3% | carey, jade, replace, replacing, aa, step, fills, phoenix |
| 61 | 23 | 0.3% | withdraws, individual, loser, app, reviews, entrant, snap, allaround |
| 62 | 22 | 0.3% | osaka, naomi, pressures, highlight, untenable, ronay, barney, big |
| 63 | 22 | 0.3% | outpouring, support, thanks, love, apos, grateful, fans, amp |
| 64 | 22 | 0.3% | nassar, larry, abuse, officials, abused, abandoned, scandal, sexual |
| 65 | 21 | 0.3% | phelps, michael, overwhelming, relate, broke, heart, says, situation |
| 66 | 21 | 0.3% | sexualisation, suits, stand, germany, body, german, wearing, fight |
| 67 | 21 | 0.3% | showdowns, hours, ladies, women, pm, triathlon, underway, supposed |
| 68 | 17 | 0.2% | covid, pole, cdc, verbal, kendricks, sam, wins, test |
| 69 | 17 | 0.2% | boyfriend, jonathan, owens, celebrates, courage, strength, supports, reunites |
| 70 | 17 | 0.2% | attitudes, superhuman, change, health, sport, callum, bbc, mental |
| 71 | 16 | 0.2% | leotards, bikini, worn, cut, decades, female, wore, objectified |
| 72 | 16 | 0.2% | weight, shoulders, feels, world, ahead, admits, feeling, constantly |
```


## 9. Consumer Interest Screening

```text
| Topic | Documents | Proxy Share | Categories | Terms |
| --- | --- | --- | --- | --- |
| 71 | 16 | 1.000 | buying, using, apparel | leotards, bikini, worn, cut, decades, female, wore, objectified |
| 22 | 53 | 0.906 | using, apparel | sexualization, unitards, wears, tired, comfort, german, promote, stand |
| 24 | 53 | 0.736 | wanting, buying, using, apparel | german, unitards, germany, wore, length, norwegian, significance, statement |
| 40 | 39 | 0.487 | searching, wanting, viewing | peacock, stream, streaming, nbc, channel, site, begins, et |
| 66 | 21 | 0.286 | using, apparel | sexualisation, suits, stand, germany, body, german, wearing, fight |
| 27 | 46 | 0.217 | using, apparel | suits, body, german, wears, opted, opt, bbc, promote |
| 50 | 28 | 0.143 | wanting, apparel | trampoline, trampolinegymnastics, knew, mobot, slaloms, skiing, solved, unlocked |
| 18 | 60 | 0.133 | wanting, participation | career, gt, couldve, years, kept, maybe, old, parents |
| 14 | 67 | 0.075 | buying, participation | health, mental, intelligence, lesson, emotional, decision, protect, message |
| 15 | 67 | 0.075 | buying, using, apparel | arms, hugging, hairdryer, best, scary, amazed, unreal, man |
| 52 | 28 | 0.071 | viewing | schedule, blog, july, day, softball, busy, ariake, centre |
| 17 | 63 | 0.063 | buying, participation | mental, category, ang, worthy, na, gold, witness, mentalhealth |
```

```text
Candidate topics nominated: 8 -> [71, 22, 24, 40, 66, 27, 50, 18]

TOPIC 71: leotards, bikini, worn, cut, decades, female, wore, objectified
   For decades, female gymnasts have worn bikini-cut leotards. In qualifying on Sunday, howev
   For decades, female gymnasts have worn bikini-cut leotards. But the German Olympics team w
   “For decades, female gymnasts have worn bikini-cut leotards. Male gymnasts wear comparativ
   For decades, female gymnasts have worn bikini-cut leotards.   The German team instead wore

TOPIC 22: sexualization, unitards, wears, tired, comfort, german, promote, stand
   Gymnastics team, tired of 'sexualization,' wears unitards https://t.co/8emTP37rNq
   German Olympic gymnastics team, tired of 'sexualization,' wears unitards - https://t.co/p5
   Gymnastics team, tired of 'sexualization,' wears unitards (from @AP) https://t.co/1AL9FBIv
   German Olympic gymnastics team opts for unitards in defiance of 'sexualization' of women h

TOPIC 24: german, unitards, germany, wore, length, norwegian, significance, statement
   I applaud Team Germany women’s gymnastics for rocking the unitard!  https://t.co/OASQEl3vl
   Germany’s women’s gymnastics team made a statement the minute they stepped into the venue 
   German gymnasts make a statement with unitards  https://t.co/ubyWSmpam2
   The Christian Science Monitor | All stories: Why German women gymnasts wore unitards at To

TOPIC 40: peacock, stream, streaming, nbc, channel, site, begins, et
   Yo I really need to find somewhere on YouTube to watch gymnastics at the olympics live 😭 h
   I don't think gymnastics is live but CBC is playing #SimoneBiles  #olympics
   @nbc next olympics can you add streaming bundles? I'd pay a decent amount to watch gymnast
   @codekittyio Hi! We are happy to help! You can see live coverage of Women's Gymnastics on 

TOPIC 66: sexualisation, suits, stand, germany, body, german, wearing, fight
   In a stand against sexualisation, Germany's female gymnasts wear full-body suits at Tokyo 
   SBS Morning News - In a stand against sexualisation, Germany's female gymnasts wear full-b
   'German gymnasts bring bodysuits to Olympic competition to fight "sexualisation" of athlet
   In a stand against sexualisation, Germany's female gymnasts wear-full body suits at Tokyo 

TOPIC 27: suits, body, german, wears, opted, opt, bbc, promote
   New: Tokyo Olympics: German women's gymnastics team wears full-body suits https://t.co/oSp
   The German women's gymnastics team opted to wear full-body suits in qualifications at the 
   🚨 German gymnastics team opt for full-body suits to promote freedom of choice #Olympics @b
   German gymnastics team opt for full-body suits to promote freedom of choice https://t.co/W

TOPIC 50: trampoline, trampolinegymnastics, knew, mobot, slaloms, skiing, solved, unlocked
   Defending @Olympics Women's #TrampolineGymnastics Individual champion @RosieMacLennan is t
   Replace Gymnastics with Juggling in the next Olympics. Problem solved.
   I just found out about Trampoline Gymnastics. Its crazy…and dangerous. #Olympics
   The gymnasts have such pretty leotards for the trampoline event. #Olympics

TOPIC 18: career, gt, couldve, years, kept, maybe, old, parents
   Maybe I could have been an #Olympics gymnast too! 🤸🏻‍♂️   In 1912.  For #TeamSweden https:
   Everyday I think about what would've happened if I kept seriously training in gymnastics a
   I have a bad back, clearly in my last life I was an @Olympics gymnast, Jesus!
   I thought I qualified for the Olympics gymnastics section because I could lift my foot abo
```


## 10. Temporal Prevalence

```text
|  | 71 | 22 | 24 | 40 | 66 | 27 | 50 | 18 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2021-07-25 | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN |
| 2021-07-26 | 0.0084 | 0.0347 | 0.0215 | 0.0108 | 0.0108 | 0.0263 | 0.0000 | 0.0144 |
| 2021-07-27 | 0.0045 | 0.0130 | 0.0156 | 0.0149 | 0.0052 | 0.0117 | 0.0000 | 0.0084 |
| 2021-07-28 | 0.0000 | 0.0000 | 0.0010 | 0.0015 | 0.0015 | 0.0020 | 0.0000 | 0.0040 |
| 2021-07-29 | 0.0000 | 0.0038 | 0.0019 | 0.0038 | 0.0000 | 0.0038 | 0.0038 | 0.0019 |
| 2021-07-30 | 0.0000 | 0.0000 | 0.0018 | 0.0018 | 0.0018 | 0.0000 | 0.0253 | 0.0054 |
| 2021-07-31 | 0.0012 | 0.0000 | 0.0023 | 0.0012 | 0.0000 | 0.0000 | 0.0047 | 0.0163 |
| 2021-08-01 | 0.0016 | 0.0016 | 0.0082 | 0.0000 | 0.0000 | 0.0000 | 0.0131 | 0.0115 |
| 2021-08-02 | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN |
| 2021-08-03 | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN |
| 2021-08-04 | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN |
| 2021-08-05 | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN |
| 2021-08-06 | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN |
| 2021-08-07 | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN |
| 2021-08-08 | 0.0000 | 0.0048 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0096 |
```

*[figure]*

*[figure]*

*[figure]*

*[figure]*

*[figure]*

*[figure]*

*[figure]*

*[figure]*


## 11. Expanding-Window Validation

```text
2021-07-26:   838 documents,   4 topics, visible: [22, 24, 27, 18]
2021-07-27: 2,379 documents,  21 topics, visible: [71, 22, 24, 40, 66, 27, 18]
2021-07-28: 4,367 documents,  29 topics, visible: [71, 22, 24, 40, 66, 27, 18]
2021-07-29: 4,896 documents,  41 topics, visible: [71, 22, 24, 40, 66, 27, 18]
2021-07-30: 5,450 documents,  65 topics, visible: [71, 22, 24, 40, 66, 27, 18]
2021-07-31: 6,310 documents,  64 topics, visible: [22, 24, 40, 66, 27, 50, 18]
2021-08-01: 6,921 documents,  67 topics, visible: [71, 22, 24, 40, 66, 27, 50, 18]
2021-08-08: 7,129 documents,  73 topics, visible: [71, 22, 24, 40, 66, 27, 50, 18]
```

```text
| Cut-off | Documents | Topics | Candidates Visible | Concentration |
| --- | --- | --- | --- | --- |
| 2021-07-25 | 2 | — | — | — |
| 2021-07-26 | 838 | 4 | [22, 24, 27, 18] | {22: np.float64(1.0), 24: np.float64(0.94), 27: np.float64(1.0), 18: np.float64(1.0)} |
| 2021-07-27 | 2,379 | 21 | [71, 22, 24, 40, 66, 27, 18] | {71: np.float64(0.93), 22: np.float64(1.0), 24: np.float64(0.98), 40: np.float64(0.91), 66: np.float64(1.0), 27: np.float64(1.0), 18: np.float64(0.76)} |
| 2021-07-28 | 4,367 | 29 | [71, 22, 24, 40, 66, 27, 18] | {71: np.float64(0.93), 22: np.float64(0.88), 24: np.float64(0.95), 40: np.float64(0.97), 66: np.float64(0.9), 27: np.float64(1.0), 18: np.float64(1.0)} |
| 2021-07-29 | 4,896 | 41 | [71, 22, 24, 40, 66, 27, 18] | {71: np.float64(0.93), 22: np.float64(0.92), 24: np.float64(0.87), 40: np.float64(0.97), 66: np.float64(0.95), 27: np.float64(0.98), 18: np.float64(1.0)} |
| 2021-07-30 | 5,450 | 65 | [71, 22, 24, 40, 66, 27, 18] | {71: np.float64(0.93), 22: np.float64(0.67), 24: np.float64(0.93), 40: np.float64(0.95), 66: np.float64(0.9), 27: np.float64(0.91), 50: 0.0, 18: np.float64(0.65)} |
| 2021-07-31 | 6,310 | 64 | [22, 24, 40, 66, 27, 50, 18] | {71: 0.0, 22: np.float64(0.92), 24: np.float64(0.96), 40: np.float64(0.95), 66: np.float64(0.86), 27: np.float64(0.96), 50: np.float64(0.95), 18: np.float64(0.88)} |
| 2021-08-01 | 6,921 | 67 | [71, 22, 24, 40, 66, 27, 50, 18] | {71: np.float64(0.88), 22: np.float64(0.94), 24: np.float64(0.96), 40: np.float64(0.95), 66: np.float64(0.95), 27: np.float64(1.0), 50: np.float64(0.96), 18: np.float64(0.9)} |
| 2021-08-08 | 7,129 | 73 | [71, 22, 24, 40, 66, 27, 50, 18] | {71: np.float64(1.0), 22: np.float64(1.0), 24: np.float64(1.0), 40: np.float64(1.0), 66: np.float64(1.0), 27: np.float64(1.0), 50: np.float64(1.0), 18: np.float64(1.0)} |
```

```text
| Topic | Retrospective | Prospective | Lag (days) |
| --- | --- | --- | --- |
| 71 | 2021-07-26 | 2021-07-27 | 1 |
| 22 | 2021-07-26 | 2021-07-26 | 0 |
| 24 | 2021-07-26 | 2021-07-26 | 0 |
| 40 | 2021-07-26 | 2021-07-27 | 1 |
| 66 | 2021-07-26 | 2021-07-27 | 1 |
| 27 | 2021-07-26 | 2021-07-26 | 0 |
| 50 | 2021-07-26 | 2021-07-31 | 5 |
| 18 | 2021-07-26 | 2021-07-26 | 0 |
```

```text
Mean detectability lag: 1.0 days (range 0 to 5)
```


## 12. Onset Assessment

```text
| Topic | First usable value | Maximum | Onset |
| --- | --- | --- | --- |
| 71 | 0.008400 | 0.008400 | indeterminate |
| 22 | 0.034700 | 0.034700 | indeterminate |
| 24 | 0.021500 | 0.021500 | indeterminate |
| 40 | 0.010800 | 0.014900 | indeterminate |
| 66 | 0.010800 | 0.010800 | indeterminate |
| 27 | 0.026300 | 0.026300 | indeterminate |
| 50 | 0.000000 | 0.025300 | 2021-07-30 |
| 18 | 0.014400 | 0.016300 | indeterminate |
```

```text
Onset observable for 1 of 8 candidates
```


## 13. External Comparison

```text
| Topic | Query | Type | Geography | Language | Aggregation | Baseline | Criterion |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 71 | leotards | search term | worldwide | en-US | daily | 2021-06-10 to 2021-07-24 | baseline mean + 1.5 SD |
| 22 | unitards | search term | worldwide | en-US | daily | 2021-06-10 to 2021-07-24 | baseline mean + 1.5 SD |
| 24 | unitards | search term | worldwide | en-US | daily | 2021-06-10 to 2021-07-24 | baseline mean + 1.5 SD |
| 40 | /g/11h582cpmh | entity | worldwide | en-US | daily | 2021-06-10 to 2021-07-24 | baseline mean + 1.5 SD |
| 66 | wearing | search term | worldwide | en-US | daily | 2021-06-10 to 2021-07-24 | baseline mean + 1.5 SD |
| 27 | suits | search term | worldwide | en-US | daily | 2021-06-10 to 2021-07-24 | baseline mean + 1.5 SD |
| 50 | trampoline | search term | worldwide | en-US | daily | 2021-06-10 to 2021-07-24 | baseline mean + 1.5 SD |
| 18 | career | search term | worldwide | en-US | daily | 2021-06-10 to 2021-07-24 | baseline mean + 1.5 SD |
```

```text
| Topic | Query | Validity | Assessment |
| --- | --- | --- | --- |
| 71 | leotards | valid | denotes competitive gymnastics apparel |
| 22 | unitards | valid | denotes the specific garment discussed in the topic |
| 24 | unitards | valid | as above; duplicates the query for topic 22, so the two do not constitute independent observations |
| 40 | /g/11h582cpmh | valid | Google Trends entity for the streaming service, disambiguated from the bird |
| 66 | wearing | invalid | generic verb, no consumer concept |
| 27 | suits | invalid | denotes business attire and legal proceedings more frequently than gymnastics apparel |
| 50 | trampoline | ambiguous | denotes both the apparatus as a consumer product and the Olympic discipline |
| 18 | career | invalid | generic term, unrelated to any consumer concept |
```

```text
Conceptually valid queries: [71, 22, 24, 40]

External correspondence, valid queries only: 4/4
Topics with ambiguous or invalid queries are excluded: a series exceeding
a threshold does not evidence correspondence where the query does not
denote the concept represented by the topic.
```


## 13.1 Onset status of the qualifying topic

```text
Topic 40 Trends onset: 2021-07-25
Topic 40 Twitter onset: None
```


## 13.2 Paired visualisation

*[figure]*

*[figure]*

*[figure]*

*[figure]*

*[figure]*

*[figure]*

*[figure]*

*[figure]*


## 14. Lead or Lag Assessment

```text
| Topic | Query | Query Validity | Twitter Onset | Trends Onset | Lead (days) | Interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| 71 | leotards | valid | None | 2021-07-25 | nan | Twitter onset indeterminate |
| 22 | unitards | valid | None | 2021-07-25 | nan | Twitter onset indeterminate |
| 24 | unitards | valid | None | 2021-07-25 | nan | Twitter onset indeterminate |
| 40 | /g/11h582cpmh | valid | None | 2021-07-25 | nan | Twitter onset indeterminate |
| 66 | wearing | invalid | None | 2021-07-26 | nan | Twitter onset indeterminate |
| 27 | suits | invalid | None | 2021-07-25 | nan | Twitter onset indeterminate |
| 50 | trampoline | ambiguous | 2021-07-30 | 2021-07-25 | -5.000000 | Trends earlier |
| 18 | career | invalid | None | 2021-08-04 | nan | Twitter onset indeterminate |
```

```text
Lead or lag estimated for 1 of 8 candidates
   Topic 50 (trampoline): -5 days — Trends earlier

Of these, 0 derive from a conceptually valid query.
The two onset criteria are not strictly commensurable: the social media
criterion is relative to a topic's own maximum, the external criterion to
the variability of a prior baseline period.
```


## 15. Published Baseline: BERTrend

```text
embedding corpus for slice-level analysis
2021-07-25: 2 documents, slice below threshold
2021-07-26:   836 documents,  67 slice topics,  15 tracked
2021-07-27: 1,541 documents, 112 slice topics,  26 tracked
2021-07-28: 1,988 documents, 135 slice topics,  43 tracked
2021-07-29:   529 documents,  42 slice topics,  46 tracked
2021-07-30:   554 documents,  42 slice topics,  50 tracked
2021-07-31:   860 documents,  57 slice topics,  53 tracked
2021-08-01:   611 documents,  48 slice topics,  56 tracked
2021-08-08:   208 documents,   9 slice topics,  58 tracked

Tracked topics: 58
```


## 15.1 Signal classification

```text
| signal | noise | strong | weak |
| --- | --- | --- | --- |
| day |  |  |  |
| 2021-07-26 | 0 | 7 | 8 |
| 2021-07-27 | 4 | 12 | 10 |
| 2021-07-28 | 9 | 23 | 11 |
| 2021-07-29 | 16 | 26 | 4 |
| 2021-07-30 | 19 | 25 | 6 |
| 2021-07-31 | 20 | 25 | 8 |
| 2021-08-01 | 20 | 27 | 9 |
| 2021-08-08 | 6 | 29 | 23 |
```


## 15.2 Comparison

```text
| Topic | BERTrend Topic | Similarity | BERTrend Flagged | Framework Detection | Difference (days) | Framework Onset |
| --- | --- | --- | --- | --- | --- | --- |
| 71 | 10 | 0.972000 | 2021-07-26 | 2021-07-27 | -1 | indeterminate |
| 22 | 7 | 0.947000 | 2021-07-26 | 2021-07-26 | 0 | indeterminate |
| 24 | 1 | 0.822000 | 2021-07-26 | 2021-07-26 | 0 | indeterminate |
| 40 | 23 | 0.916000 | 2021-07-27 | 2021-07-27 | 0 | indeterminate |
| 66 | 1 | 0.875000 | 2021-07-26 | 2021-07-27 | -1 | indeterminate |
| 27 | 1 | 0.997000 | 2021-07-26 | 2021-07-26 | 0 | indeterminate |
| 50 | 0 | 0.779000 | 2021-07-26 | 2021-07-31 | -5 | 2021-07-30 |
| 18 | 0 | 0.812000 | 2021-07-26 | 2021-07-26 | 0 | indeterminate |
```

```text
Recovered by BERTrend          : 8 of 8
Flagged on first usable date   : 7
Framework onset indeterminate  : 7
```


## 16. Two-Coder Qualification

```text
'rng = np.random.default_rng(seed)\n\nsample_rows = []\nfor t in candidates:\n    topic_docs = gym[gym["topic"] == t]\n    for _, r in topic_docs.sample(min(15, len(topic_docs)),\n                                  random_state=seed).iterrows():\n        sample_rows.append({"topic": t, "text": r["text"], "screen_flagged": 1})\n\nnon_candidates = [t for t in topic_words if t not in candidates]\nfor t in rng.choice(non_candidates, size=min(10, len(non_candidates)),\n                    replace=False):\n    topic_docs = gym[gym["topic"] == t]\n    for _, r in topic_docs.sample(min(3, len(topic_docs)),\n                                  random_state=seed).iterrows():\n        sample_rows.append({"topic": int(t), "text": r["text"], "screen_flagged": 0})\n\ncoding_sheet = (pd.DataFrame(sample_rows)\n                .sample(frac=1, random_state=seed)\n                .reset_index(drop=True))\ncoding_sheet["rater_1"] = ""\ncoding_sheet["rater_2"] = ""\n\ncoding_sheet.to_csv(out_folder / "coding_sheet.csv", index=False)\n\n# mentions and from descriptions of athletes\' behaviour\ninstructions = """CODING INSTRUCTIONS\n\nLabel each document 2, 1 or 0.\n\n2 - CONSUMER INTEREST\n    The author expresses their own searching, wanting, buying, using, price or\n    brand interest, viewing, or participation.\n      "where can I buy one of those unitards"\n      "next olympics can you add streaming bundles? I\'d pay a decent amount"\n      "thinking of signing my daughter up for gymnastics classes"\n\n1 - PRODUCT MENTION\n    A product, brand or service is present, but the author expresses no\n    consumer engagement with it.\n      "those unitards are really well designed"\n      "the Nike leotards look different this year"\n\n0 - DESCRIPTION OR REPORTING\n    Describes an event, athlete behaviour, competition result or news.\n      "German Olympic gymnastics team, tired of \'sexualization,\' wears unitards"\n      "For decades, female gymnasts have worn bikini-cut leotards"\n      "Hashimoto wins gold in the all-around final"\n\nThe distinction between 2 and 0 is whether the AUTHOR engages as a consumer or\nreports what someone else did. A headline describing what athletes wore is 0,\nhowever much product vocabulary it contains.\n\nCode independently. Do not discuss while coding. Do not sort the file.\n"""\n\n(out_folder / "coding_instructions.txt").write_text(instructions)\n\nprint(f"Coding instrument: {len(coding_sheet)} documents")\nprint(f"   {out_folder}/coding_sheet.csv")\nprint(f"   {out_folder}/coding_instructions.txt")'
```


## 16.1 Agreement, screen evaluation and qualification

```text
Documents coded        : 150
Cohen's kappa (linear) : 0.461
Raw agreement          : 87.3%

Agreed label distribution
   2 consumer interest         : 5
   1 product mention           : 4
   0 description or reporting  : 122

Agreement by category
   2 consumer interest         :   5 of  15 (33%) where either coder used the label
   1 product mention           :   4 of  14 (29%) where either coder used the label
   0 description or reporting  : 122 of 140 (87%) where either coder used the label

Screen evaluated against 131 agreed labels
   precision 0.05   recall 1.00   F1 0.09
```

```text
| Topic | Coded | Consumer interest | Product mention | Proportion | Qualifies |
| --- | --- | --- | --- | --- | --- |
| 71 | 14 | 0 | 2 | 0.000000 | No |
| 22 | 13 | 0 | 0 | 0.000000 | No |
| 24 | 10 | 0 | 2 | 0.000000 | No |
| 40 | 13 | 5 | 0 | 0.380000 | Yes |
| 66 | 14 | 0 | 0 | 0.000000 | No |
| 27 | 15 | 0 | 0 | 0.000000 | No |
| 50 | 15 | 0 | 0 | 0.000000 | No |
| 18 | 9 | 0 | 0 | 0.000000 | No |
```

```text
Qualifying topics: [40]
```


## 16.2 Disagreement analysis

```text
19 disagreements of 150 documents

Disagreement pattern by label pair
rater_1  rater_2
0        1          3
         2          7
1        0          6
         2          1
2        0          2

Disagreements by topic
topic
18    6
24    5
22    2
40    2
54    2
66    1
71    1

Examples

[0 vs 2] topic 18: Everyday I think about what would've happened if I kept seriously training in gymnastics and di

[1 vs 0] topic 22: Love seeing more of this
#Feminism
#SportsNotSexualization
Wearing unitards, German gymnasts pr

[1 vs 0] topic 22: #Tokyo #Olympics2020: #German team debuts unitards to protest ‘#sexualisation in #gymnastics’



[0 vs 2] topic 18: There's a 32 year old competing in the gymnastics final and so I have now convinced myself that

[1 vs 0] topic 24: Racism for the Gold! 

@Olympics allowed the German gymnastics team to debut a full-length leot

[1 vs 0] topic 24: @catnip In not quite related news, I hope this decision by Germany's gymnasts is the start of s

[1 vs 0] topic 24: germany's women's gymnastics team and norway's women's beach volleyball team are really out her

[0 vs 2] topic 18: 11 years old me watching the Olympics: artistic gymnastics is beautiful I want to be a gymnast🥺
```


## 16.3 Sensitivity of the qualification criteria

```text
EFFECT OF EXCLUDING DISAGREEMENTS ON TOPIC PROPORTIONS

 Topic   Sampled   Agreed  Dropped   % dropped   Proportion
------------------------------------------------------------
    71        15       14        1          7%         0.00
    22        15       13        2         13%         0.00
    24        15       10        5         33%         0.00
    40        15       13        2         13%         0.38
    66        15       14        1          7%         0.00
    27        15       15        0          0%         0.00
    50        15       15        0          0%         0.00
    18        15        9        6         40%         0.00


THRESHOLD VARIATION

threshold 0.05: [40]
threshold 0.10: [40]
threshold 0.15: [40]
threshold 0.20: [40]
threshold 0.25: [40]  <- adopted
threshold 0.30: [40]
threshold 0.33: [40]
threshold 0.40: none
threshold 0.50: none

Topic 40 proportion under agreed labels: 0.38


DISAGREEMENT SENSITIVITY

Rater 1 only        : qualifies [40]          (24:0.07, 40:0.33, 66:0.07)
Rater 2 only        : qualifies [40, 18]      (40:0.47, 18:0.40)
Agreed only         : qualifies [40]          (40:0.38)
Optimistic (max)    : qualifies [40, 18]      (24:0.07, 40:0.47, 66:0.07, 18:0.40)
Conservative (min)  : qualifies [40]          (40:0.33)

Qualifies under every rule : [40]
Qualifies under some only  : [18]
```


## 16.4 Generalisability of the screen evaluation

```text
Agreed labels               : 131
Consumer interest cases     : 5
Documents flagged by screen : 103
True positives              : 5

Precision 0.05   95% CI [0.02, 0.11]
Recall    1.00   95% CI [0.57, 1.00]
```


## 17.1 Data sufficiency threshold

```text
threshold  10: excludes ['2021-07-25']
threshold  20: excludes ['2021-07-25']
threshold  30: excludes ['2021-07-25']
threshold  50: excludes ['2021-07-25']
```


## 17.2 Onset criterion

```text
fraction 0.10: onset observable for [50]
fraction 0.25: onset observable for [50]
fraction 0.40: onset observable for [50]
fraction 0.50: onset observable for [50]
```


## 17.3 External change criterion

```text
k=1.0: {71: datetime.date(2021, 7, 25), 22: datetime.date(2021, 7, 25), 24: datetime.date(2021, 7, 25), 40: datetime.date(2021, 7, 25)}
k=1.5: {71: datetime.date(2021, 7, 25), 22: datetime.date(2021, 7, 25), 24: datetime.date(2021, 7, 25), 40: datetime.date(2021, 7, 25)}
k=2.0: {71: datetime.date(2021, 7, 25), 22: datetime.date(2021, 7, 25), 24: datetime.date(2021, 7, 25), 40: datetime.date(2021, 7, 25)}
k=2.5: {71: datetime.date(2021, 7, 25), 22: datetime.date(2021, 7, 25), 24: datetime.date(2021, 7, 25), 40: datetime.date(2021, 7, 25)}
```


## 18. Summary

```text
| Assessment | Metric | Value |
| --- | --- | --- |
| RQ1 | Topics identified | 73 |
| RQ1 | Coherence (UMass) | -2.097000 |
| RQ1 | Topic diversity | 0.844000 |
| RQ1 | Seed consistency | 0.633000 |
| RQ1 | Outlier share | 37.6% |
| RQ1 | Candidate topics | 8 |
| RQ1 | Qualifying topics | 1 |
| RQ2 | External correspondence (valid queries) | 4/4 |
| RQ3 | Onset observable | 1/8 |
| RQ3 | Lead or lag estimated | 1/8 |
| RQ3 | Lead or lag from a valid query | 0/8 |
| Validation | Mean detectability lag (days) | 1.000000 |
| Validation | Candidates recovered by BERTrend | 8/8 |
```

```text
12 artefacts written to the project outputs folder
```

