# Text Recognition Model

## ChineseOCR

### Original Network Link

https://github.com/HangZhouShuChengKeJi/chinese-ocr

### pb Model Link:

Instructions: https://bbs.huaweicloud.com/forum/thread-77538-1-1.html

### Convert model To Ascend om file

In order to improve the inference efficiency, we can set the multiple batches **n**
```bash
atc --model=./chineseocr.pb \
    --framework=3 \
    --output=./chineseocr_32_320_n_batch \
    --soc_version=Ascend310 \
    --insert_op_conf=./chineseocr_aipp.cfg \
    --input_shape="the_input:n,32,320,1"
```

## CRNN

### Original Network Link

https://github.com/MaybeShewill-CV/CRNN_Tensorflow

### pb Model Link:

https://bbs.huaweicloud.com/forum/thread-85713-1-1.html

### Convert model To Ascend om file

In order to improve the inference efficiency, we can set the multiple batches **n**
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
**enable_scope_fusion_passes** specifies the fusion rules at compile time

## Products that have been verified:

- Atlas 800 (Model 3000)
- Atlas 800 (Model 3010)
- Atlas 300 (Model 3010)