# 分类模型

## 原始模型参考论文地址

https://arxiv.org/abs/1512.03385

## 模型下载地址

https://www.huaweicloud.com/ascend/resources/modelzoo/Models/7548422b6b9c4a809114435f6b128bb6

在该地址下载pb文件``resnet_v1_50.pb``

### 模型后处理标签文件(imagenet1000_clsidx_to_labels) 

https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt

下载后放置在目录resent下

## 依赖

可根据需求修改模型的aipp配置文件

https://support.huawei.com/enterprise/zh/doc/EDOC1100150964/15bb57f6

## 转换tensorflow模型至昇腾om模型

```bash
cd resnet
atc --model=./resnet_v1_50.pb \
    --framework=3 \
    --output=./resnet50_aipp \
    --soc_version=Ascend310 \
    --insert_op_conf=./aipp_resnet50.cfg \
    --input_shape="input:1,224,224,3"
```

## 转换单算子模型至昇腾om模型

```bash
cd ..
atc --singleop single_op/op_list.json \
    --output ./single_op \
    --soc_version=Ascend310
```

注意：```soc_version```需要指定对应的昇腾AI处理器的版本

## 已验证的产品

- Atlas 800 (Model 3000)
- Atlas 800 (Model 3010)
- Atlas 300 (Model 3010)