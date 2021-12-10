# 文字识别模型

## ChineseOCR

### 原始模型网络介绍地址

https://github.com/HangZhouShuChengKeJi/chinese-ocr

### pb模型获取方法

参考地址 https://bbs.huaweicloud.com/forum/thread-77538-1-1.html

### 转换模型至昇腾om模型

为了提升推理效率，可以使用多batch推理，其中**n**表示batch的数目
```bash
atc --model=./chineseocr.pb \
    --framework=3 \
    --output=./chineseocr_32_320_n_batch \
    --soc_version=Ascend310 \
    --insert_op_conf=./chineseocr_aipp.cfg \
    --input_shape="the_input:n,32,320,1"
```


## CRNN

### 原始模型网络介绍地址

https://github.com/MaybeShewill-CV/CRNN_Tensorflow

### pb模型获取方法

https://bbs.huaweicloud.com/forum/thread-85713-1-1.html

### 转换模型至昇腾om模型

为了提升推理效率，可以使用多batch推理，其中**n**表示batch的数目
```bash
atc --model=./crnn.pb \
    --framework=3 \
    --output=./crnn_chinese_32_320_n_batch \
    --enable_scope_fusion_passes=ScopeDynamicRNNPass \
    --out_nodes=shadow_net/sequence_rnn_module/raw_prediction:0 \
    --soc_version=Ascend310 \
    --insert_op_conf=./crnn_aipp.cfg \
    --input_shape="input_tensor:n,32,320,3"
```
**enable_scope_fusion_passes**为指定编译时需要生效的融合规则

## 已验证的产品

- Atlas 800 (Model 3000)
- Atlas 800 (Model 3010)
- Atlas 300 (Model 3010)