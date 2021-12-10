# 目标检测模型

## 原始模型网络介绍地址

https://github.com/ChenYingpeng/caffe-yolov3

## 模型下载地址

### Caffe

https://pan.baidu.com/s/1yiCrnmsOm0hbweJBiiUScQ

在该地址下载prototxt配置文件``yolov3_416.prototxt``和模型权重文件``yolov3_416.caffemodel``

### Tensorflow

https://www.huaweicloud.com/ascend/resources/modelzoo/Models/5cb8d2255070439d94068b7935570c43

在该地址下载模型pb文件
参考地址 https://bbs.huaweicloud.com/forum/thread-72656-1-1.html

## 依赖

可根据需求修改模型的prototxt配置文件和aipp配置文件

### 修改prototxt配置文件
**如果模型为Caffe，必须参考下面的链接修改原始的prototxt文件，否则直接运行该sample会出错**

https://support.huawei.com/enterprise/zh/doc/EDOC1100150964/3502a21a#ZH-CN_TOPIC_0257009563

### 修改aipp配置文件

https://support.huawei.com/enterprise/zh/doc/EDOC1100150964/15bb57f6

## 转换模型至昇腾om模型

### Caffe
```bash
cd yolov3
atc --model=./yolov3.prototxt \
    --weight=./yolov3.caffemodel \
    --framework=0 \
    --output=./yolov3_aipp \
    --soc_version=Ascend310 \
    --insert_op_conf=./aipp_yolov3.cfg
```

### Tensorflow
```bash
atc --model=./yolov3_tf.pb \
    --framework=3 \
    --output=./yolov3_aipp \
    --input_shape="input/input_data:1,416,416,3" \
    --out_nodes="conv_lbbox/BiasAdd:0;conv_mbbox/BiasAdd:0;conv_sbbox/BiasAdd:0" \
    --insert_op_conf=./aipp_yolov3.cfg \
    --soc_version=Ascend310
```

注意：```soc_version```需要指定对应的昇腾AI处理器的版本

## 模型更换

如果要更换其他输入规格的YoloV3模型，需修改模型相关的配置信息：

### 修改setup.config配置文件
./data/config/setup.config
```bash
model_width = xxx
model_height = xxx
```
### 修改aipp配置文件
./data/model/yolov3/aipp_yolov3.cfg
```bash
src_image_size_w : xxx
src_image_size_h : xxx
```
### 修改yolov3.prototxt文件(若模型为Caffe)
```bash
input_shape {
  dim: 1
  dim: 3
  dim: xxx
  dim: xxx
}
```
其中xxx为新模型支持的输入图片的宽和高

## 已验证的产品

- Atlas 800 (Model 3000)
- Atlas 800 (Model 3010)
- Atlas 300 (Model 3010)