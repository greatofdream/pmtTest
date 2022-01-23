# PMT-TEST
## CODE STRUCTURE
+ `documentation`: record the process of test.
+ `waveana`: 波形处理分析的基类，拓展为波形分析和带触发波形分析。

## Feature
+ 基础参数测量流程：直接分析每个波形中的电荷分布
  + `Charge`
  + `Gain`
  + `P/V`
  + `Resolution`
  + `DarkNoise Rate`
+ 高级参数测量流程：分析激光触发后的时间窗(基础参数测量给出)->根据窗口选择出触发后的波形参数
  + `Charge`
  + `Gain`
  + `P/V`
  + `Resolution`
  + `DE`
  + `TR`,`TF`,`TH`
  + `Prompt pulse`, `Delay pulse`
  + `TTS`

## Example