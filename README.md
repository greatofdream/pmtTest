# Introduction
```
make sk6_summary.csv -f initRuns.mk
make sk7_summary.csv -f initRuns.mk
```
+ fix the typo error:
  + `86157`: `X=12m`->`X=-12m`
```
make phase1=sk6 phase2=sk7 -f initRuns.mk sk6_sk7.csv
make phase1=sk6 phase2=sk7 -f initRuns.mk sk6_sk7_ZE.csv
```

```
python3 ana.py
```

```
python3 compare.py -i sk6_sk7_ZE.csv
python3 compare.py -i sk6_sk6.csv --onebyone
```
