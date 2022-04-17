# PMT-TEST
## CODE STRUCTURE
+ `documentation`: record the process of test.
+ `waveana`: 波形处理分析的基类，拓展为波形分析和带触发波形分析。

## Feature
+ 基础参数测量流程：直接分析每个波形中最大峰的电荷分布
  + `Charge`, `Gain`,`P/V`,`Resolution`,`DarkNoise Rate`
  + `spePreAnalysisUproot3.py`:直接分析
+ 高级参数测量流程：分析激光触发后的时间窗(基础参数测量给出)->根据窗口选择出触发后的波形参数
  + `Charge`,`Gain`,`P/V`,`Resolution`,`DE`,`TR`,`TF`,`TH`,`Prompt pulse`, `Delay pulse`,`TTS`
  + `triggerPreAnalysis.py`:考虑触发时间窗后，进行预分析
  + `summary.py`： 处理出触发后的时间窗区域
  + `triggerPulse.py`: 根据触发后的时间窗区域重新分析是否有触发 `pulseRatio.py`:分析前后脉冲比例 `pulseRatioWOnoise.py`:去除暗噪声计数的结果
  + `triggerPulse.py`: 根据触发后的时间窗区域重新分析是否有触发 `QE.py`: 分析触发比例
## Example
+ 暗噪声分析run679, channels 2和3
  ```
  make anaNum=679 channels="2 3" -f Makefiles/DN/Makefile
  ```
+ 激光触发分析run680, channels 2和3， 触发通道1
  ```
  make anaNum=680 channels="2 3" -f Makefiles/Trigger/Makefile
  ```
+ `QE`测量依赖于两次的交换测量，需要在task中创建一个`xxx/config.json`，执行
  ```
  make xxx/result.log
  ```