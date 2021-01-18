# VFX_2019S

High Dynamic Range Imaging
## Requirements
- python3
- opencv 4.0
- jupyter notebook
## Packages(pip3)
- numpy
- matplotlib
## Execute
直接輸入指令
```
$ jupyter notebook HighDynamicRangeImaging.ipynb
```
## Input
將.ipynb檔案與照片檔放在同一個資料夾，並且在同一個資料夾裡新建input.txt，寫入照片的檔名和曝光時間，格式如下：
```
#Filename	Exposure_time
DSCXXXXX.JPG	2.0
```
範例的input.txt：
```
#Filename			Exposure_time
LRG_DSC00524.JPG	0.00025
LRG_DSC00525.JPG	0.0005
LRG_DSC00526.JPG	0.001
LRG_DSC00527.JPG	0.002
LRG_DSC00528.JPG	0.004
LRG_DSC00529.JPG	0.008
LRG_DSC00530.JPG	0.017
LRG_DSC00531.JPG	0.033
LRG_DSC00532.JPG	0.067
```
## Output
一個生成的.hdr檔和數個不同的.jpg檔，可選擇呈現結果較好的
```
Result.hdr
1Result_Drago.jpg
1Result_Drago.jpg
...
```
