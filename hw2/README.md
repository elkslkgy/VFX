# VFX_2019S

Image Stitching
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
$ jupyter notebook Image Stitching.ipynb
```
## Input
在.ipynb檔案所在的資料夾加入imgtest資料夾，Result資料夾、照片檔與readImage.txt寫入照片的檔名和Focal length，格式如下：
```
#Filename     Focal_length
DSCXXXXX.JPG  800
```
範例的readImage.txt：
```
#File name        Focal length
LRG_DSC01039.JPG  900
LRG_DSC01040.JPG  900
LRG_DSC01041.JPG  900
LRG_DSC01042.JPG  900
LRG_DSC01043.JPG  900
LRG_DSC01044.JPG  900
```
## Output
Result資料夾中會有一個Stiching完成的結果與切除黑邊後的結果
