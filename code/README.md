# 文件解释

## tools  文件夹(与项目无关)

- `data.py`：加载数据的类
- `view.py`：标注图片的异常区域，观察结果的图片
- `view_cut_pic.py`：将上个代码观察的图片切成 `512 X 512` 均匀大小，并生成缺陷区域的 `mask`

## 当前文件夹

- `coco.py`：制作 `COCO` 数据的 `json` 文件
- `utils.py`：小工具
- `mask.py`：制作图片的 `mask` 数据
- `crop.py`：切分测试数据，记录切分图片的相对位置
- `cut.py`：裁剪训练数据与对应的`mask`数据
- `check_mask.py`：检验 `mask` 图像是否切割正确

