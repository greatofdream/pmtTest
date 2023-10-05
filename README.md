# Introduction
## extract the LINAC run summary file
```
make sk6_summary.csv -f initRuns.mk
make sk7_summary.csv -f initRuns.mk
```
+ fix the typo error:
  + `86157`: `X=12m`->`X=-12m`
## comparision between sk6 and sk7
+ overlay the different run in a same plots, and plot for each run
```shell
make all
```

## make LINAC run dat and Ge dat
```shell
make sk6_linac_runsum.dat sk6_ge_runsum.dat -f initRuns.mk

make sk7_linac_runsum.dat sk7_ge_runsum.dat -f initRuns.mk
```
## MC production
+ generate the data: the linac macro should be set whether using the tentative
```shell
make all
```
## Recon
### recon code
+ in `recon_src`
```
make clean
make tentative=1
make clean
make
```
## recon execution
+ For data: if use tentative parameter, `tentative_skx.txt` should be created and recon with tentative flag
```
make phase=sk6 -f recon.mk
make phase=sk7 tentative=1 -f recon.mk
```
+ For MC
```
make phase=sk6 allskg4 -f recon.mk
make phase=sk7 tentative=1 allskg4 -f recon.mk
```
## Compare
+ comparision of basic parameter distributions for each run
```
make phase=sk6 compareDataMC -f compare.mk
make phase=sk7 compareDataMC -f compare.mk
```
+ check the `N_eff` difference between Data and MC
```
make phase=sk6 run_b=086119 run_e=086161 compareResult/sk6/compare_all.pdf -f compare.mk
make phase=sk7 run_b=091368 run_e=091560 compareResult/sk7/compare_all.pdf -f compare.mk
```
## Tune COREPMT
+ skg4: `linac.mac`
```
/SKG4/Detector/ID/CorrectionFactor 0.88
```
