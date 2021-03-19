# Object Detection Model

## Original Network Link

https://github.com/ChenYingpeng/caffe-yolov3

## Pre-trained Model Link:

### Caffe

https://pan.baidu.com/s/1yiCrnmsOm0hbweJBiiUScQ

Download the prototxt configuration file ``yolov3_416.prototxt`` and the model weight file ``yolov3_416.caffemodel`` from this link

### Tensorflow

https://www.huaweicloud.com/ascend/resources/modelzoo/Models/5cb8d2255070439d94068b7935570c43

Download the pb file
Instructions: https://bbs.huaweicloud.com/forum/thread-72656-1-1.html

## Dependency
Please refer to the Developer Manual to customize the prototxt file and aipp config file

### Modify the prototxt file
**If the model is Caffe, you must modify the prototxt file by referring to the following link. Otherwise, an error will occur when running the sample**

https://support.huawei.com/enterprise/en/doc/EDOC1100150947/e5c45637/caffe-network-model#EN-US_TOPIC_0257009563

### Modify the aipp configuration file

https://support.huawei.com/enterprise/en/doc/EDOC1100150947/c57da3db/configuration-file-template

## Convert model To Ascend om file

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

Attention: ```soc_version``` needs to specify the version of the corresponding Ascend AI processor

## Model replacement

Replace the YoloV3 model with other input specifications, we need to modify the configuration as follow:

### Configure setup.config
./data/config/setup.config
```bash
model_width = xxx
model_height = xxx
```

### Configure aipp_yolov3.cfg
./data/model/yolov3/aipp_yolov3.cfg
```bash
src_image_size_w : xxx
src_image_size_h : xxx
```

### Configureyolov3.prototxt(Caffe Model)
```bash
input_shape {
  dim: 1
  dim: 3
  dim: xxx
  dim: xxx
}
```
xxx is the new model input weight and height

## Products that have been verified:

- Atlas 800 (Model 3000)
- Atlas 800 (Model 3010)
- Atlas 300 (Model 3010)