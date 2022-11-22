# PMT-TEST
[Chinese](./README_cn.md)
## CODE STRUCTURE
+ documentation: Docs of this repo
+ Makefiles: Makefile for running code
+ waveana: Core waveform analysis
+ docscript: Function tests
## Feature
Two stages in measurements: Dark noise, Laser.
+ Dark noise: period sample without laser
  + `Charge`, `Gain`,`P/V`,`Resolution`,`DarkNoise Rate`
+ Laser: trigger with laser
  + `Charge`,`Gain`,`P/V`,`Resolution`,`DE`,`TR`,`TF`,`TH`,`Prompt pulse`, `Delay pulse`,`TTS`