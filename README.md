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
  + `TriggerPreAna.py`:考虑触发时间窗后，进行预分析，为了加速，使用了600ns的长度对波形做了切割，所以这里在调整激光的上升沿时，要保证触发的信号在600ns以内，或者调整这个大小。
  + `summary.py`： 处理出触发后的时间窗区域
  + `TriggerAna.py`: 根据触发后的时间窗区域重新分析是否有触发 
  + `QERatio.py`:分析触发比例，输出结果在`qe/ratio.h5.log` 
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
+ 预览激光触发数据前100条波形
  ```
  make preview anaNum=680 channels="2 3" triggerch=1 -f Makefiles/Trigger/Makefile
  ```
  或者预览单条波形
  ```
  make ExResult/[run号]/eid[eid号].pdf anaNum=680 channels="2 3" triggerch=1 -f Makefiles/Trigger/Makefile
  ```
+ `QE`测量依赖于两次的交换测量，需要在task中创建一个`xxx/config.json`，执行
  ```
  make xxx/result.log
  ```
## 分析介绍
### 基线和均方差计算
+ 众数法
    + 对波形的幅值作直方图，取众数作为第一次基线估计值，整体均方差为第一次估计
    + 选择Base−5𝜎为阈值，选择出信号区域，并将信号区域左右延展10ns
    + 选择非信号区域，重新计算基线和均方差
+ 随机法
    + 将波形均分成10端，求每段的标准差，标准差最小的一段，取其均值作为基线值
对上述两种方法取均方差最小的结果，返回基线和均方差
### 寻峰
取每段波形中最小值
### 电荷积分
### 寻找单光电子峰
