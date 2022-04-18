# PMT-TEST
## CODE STRUCTURE
+ `documentation`: record the process of test.
+ `waveana`: 波形处理分析的基类，拓展为波形分析和带触发波形分析。

## Feature
+ 基础参数测量流程：直接分析每个波形中最大峰的电荷分布
  + `Charge`, `Gain`,`P/V`,`Resolution`,`DarkNoise Rate`
  + `DCRPreAna.py`:直接分析
  + `srcuproot3/spePrePlot.py`:绘制分析结果的分布图
  + `srcuproot3/calcTriggerRate.py`:计算暗噪声计数率DCR，输出的内容单位是kHz,在log日志中
  + `srcuproot3/waveProfile.py`:筛选出峰值附近的波形，累加并平均。
+ 高级参数测量流程：分析激光触发后的时间窗(基础参数测量给出)->根据窗口选择出触发后的波形参数
  + `Charge`,`Gain`,`P/V`,`Resolution`,`DE`,`TR`,`TF`,`TH`,`Prompt pulse`, `Delay pulse`,`TTS`
  + `TriggerPreAna.py`:考虑触发时间窗后，进行预分析
  + `summary.py`： 处理出触发后的时间窗区域
  + `triggerPulse.py`: 根据触发后的时间窗区域重新分析是否有触发 
  + `pulseRatio.py`:分析前后脉冲比例 `pulseRatioWOnoise.py`:去除暗噪声计数的结果
  + `QE.py`: 分析触发比例
## Example
+ 暗噪声分析run679, channels 2和3;根据需要调整需要的核数，比如16核`-j16`
  ```
  make anaNum=679 channels="2 3" -f Makefiles/DN/Makefile -j16
  ```
+ 激光触发分析run680, channels 2和3， 触发通道1;根据需要调整需要的核数，比如16核`-j16`
  ```
  make anaNum=680 channels="2 3" triggerch=1 -f Makefiles/Trigger/Makefile -j16
  ```
+ `QE`测量依赖于两次的交换测量，需要在task中创建一个`xxx/config.json`，执行
  ```
  make xxx/result.log
  ```
