# Classification Model

## Original Model Reference Paper Link:

https://arxiv.org/abs/1512.03385

## Pre-trained Model Link:

https://www.huaweicloud.com/ascend/resources/modelzoo/Models/7548422b6b9c4a809114435f6b128bb6

Download the pb configuration file ``resnet_v1_50.pb`` from this link

##### ImageNet lable txt(imagenet1000_clsidx_to_labels) link

https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt

Download imagenet1000_clsidx_to_labels.txt and place it under the directory resent

## Dependency

https://support.huawei.com/enterprise/en/doc/EDOC1100150947/c57da3db/configuration-file-template

## Convert tensorflow model To Ascend om file

```bash
cd resnet
atc --model=./resnet_v1_50.pb \
    --framework=3 \
    --output=./resnet50_aipp \
    --soc_version=Ascend310 \
    --insert_op_conf=./aipp_resnet50.cfg \
    --input_shape="input:1,224,224,3"
```

## Convert single op model To Ascend om file

```bash
cd ..
atc --singleop single_op/op_list.json \
    --output ./single_op \
    --soc_version=Ascend310
```

Attention: ```soc_version``` needs to specify the version of the corresponding Ascend AI processor

## Versions that have been verified:

- Atlas 800 (Model 3000)
- Atlas 800 (Model 3010)
- Atlas 300 (Model 3010)