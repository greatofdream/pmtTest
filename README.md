# PMT-TEST
## CODE STRUCTURE
+ `documentation`: record the process of test.
+ `waveana`: 波形处理分析的基类，拓展为波形分析和带触发波形分析。

## Feature
+ 基础参数测量流程：直接分析每个波形中最大峰的电荷分布
  + `Charge`, `Gain`,`P/V`,`Resolution`,`DarkNoise Rate`
  + `DCRPreAna.py`:直接分析
  + `srcuproot3/spePrePlot.py`:绘制分析结果(电荷，峰高，上升下降时间等)的分布图 `charge.h5`,`charge.h5.pdf`
  + `srcuproot3/calcTriggerRate.py`:计算暗噪声计数率DCR，输出的内容单位是kHz,在log日志中(`DN.pdf.log`)，在`DN.pdf`中包含噪声和信号区别的阈值选择cut
  + `srcuproot3/waveProfile.py`:筛选出峰值附近的波形，累加并平均得到单光电子波形。
    + 输出在`profile.h5`,`profile.h5.pdf`
    + 目前选择单光电子峰附近的[-5,+5]mVns区间的波形进行平均；此外如果所有波形没有信号，上面的`spePrePlot.py`给出的单光电子峰会出现错误，因此额外加上大于30mVns的阈值限制
+ 高级参数测量流程：分析激光触发后的时间窗(基础参数测量给出)->根据窗口选择出触发后的波形参数
  + `Charge`,`Gain`,`P/V`,`Resolution`,`DE`,`TR`,`TF`,`TH`,`Prompt pulse`, `Delay pulse`,`TTS`
  + `TriggerPreAna.py`:考虑触发时间窗后，进行预分析
  + `summary.py`： 处理出触发后的时间窗区域
  + `triggerPulse.py`: 根据触发后的时间窗区域重新分析是否有触发 
  + `pulseRatio.py`:分析前后脉冲比例 `pulseRatioWOnoise.py`:去除暗噪声计数的结果
  + `QE.py`: 分析触发比例
## Example
+ 暗噪声分析run679, channels 2和3;根据需要调整需要的核数，比如16核`-j16`;分析结果包括基础参数测量，即下面的make会自动执行上述基础参数测量中的所有程序
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
