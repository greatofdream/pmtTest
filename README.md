# PMT-TEST
[Chinese](./README_cn.md)
## CODE STRUCTURE
+ summary of PMTs: results are store in `ExPMT`
  + `combineDarkAna.py`: merge dark noise results of a PMT
  + `combineLaserAna.py`: merge dark noise results of a PMT
  + `darkLaserCompare.py`: visualize results of PMTs
+ summary of DN Runs: results are store in `ExResults`
  + `DCRPreAna.py`
  + `BasicSummary.py`
+ summary of Laser Runs: results are store in `ExResults`
  + `TriggerPreAna.py`
  + `BasicSummary.py`
  + `TrigInterval.py`
  + `TriggerAna.py`
  + `TriggerSummary.py`
  + `TriggerPulseAna.py`
  + `TriggerPulseSummary.py`
+ utils for analyze
  + `config.py`
  + `csvDatabase.py`
+ `documentation`: Docs of this repo
+ `Makefiles/`: Makefile for running code
  + `Trigger/`
  + `DN/`
+ `waveana/`: Core waveform analysis
+ `docscript/`: Function tests
+ `ExPMT/`: results of PMTs, default find all PMT in `RUNINFO.csv`
  + `ExcludePMT.csv`: excluded PMTs
  + `ExcludeRun.csv`: excluded Runs
  + `PMTSummary.csv`: summary results of PMTs
  + `Calibration.csv`: output of PDE and calibration results
  + `LowStatisticsPMT.csv`: low statistics PMT, which affect in summary for afterpulse

## Feature
Two stages in measurements: Dark noise, Laser.
+ Dark noise: period sample without laser
  + `Charge`, `Gain`,`P/V`,`Resolution`,`DarkNoise Rate`
+ Laser: trigger with laser
  + `Charge`,`Gain`,`P/V`,`Resolution`,`DE`,`TR`,`TF`,`TH`,`Prompt pulse`, `Delay pulse`,`TTS`