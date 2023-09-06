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
+ modify the SKG4 code
```c++
// src/Construct/SKG4DetectorMessenger.cc:L77
	fWaterTransparencyCmd = new G4UIcmdWithADouble("/SKG4/Detector/Material/WaterTransparency",this);
	fWaterTransparencyCmd->SetGuidance("Set specified water transparency");
	fWaterTransparencyCmd->SetParameterName("fWaterTransparency",0.);
	fWaterTransparencyCmd->SetDefaultValue(0.);
// src/Construct/SKG4DetectorMessenger.cc:L77
if (command==fWaterTransparencyCmd)	
      SKG4Materials->SetWaterTransparency(fWaterTransparencyCmd->GetNewDoubleValue(newValue));
// src/Construct/SKG4DetectorMessenger.cc:L379
delete fWaterTransparencyCmd;
// include/Construct/SKG4DetectorMessenger.cc:L65
G4UIcmdWithADouble*   fWaterTransparencyCmd;
// include/Construct/SKG4ConstructMaterials.hh:L167
G4double fWaterTransparency;
// include/Construct/SKG4ConstructMaterials.hh:L70
		inline void SetWaterTransparency(G4double val)	{fWaterTransparency = val;}
		inline G4double GetWaterTransparency() const {return fWaterTransparency;}
// src/Construct/SKG4ConstructMaterials.cc:L626
fWaterT = GetWaterTransparency();

```
+ generate the data: the linac macro should be set whether using the tentative
```shell
make phase=sk6 -f mc.mk
make phase=sk7 -f mc.mk
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
