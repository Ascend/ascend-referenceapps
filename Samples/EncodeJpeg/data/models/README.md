## Classification Model

##### Original Network Link

https://github.com/KaimingHe/deep-residual-networks/blob/master/prototxt/ResNet-50-deploy.prototxt

#### Pre-trained Model Link:

https://onedrive.live.com/?authkey=%21AAFW2-FVoxeVRck&id=4006CBB8476FF777%2117887&cid=4006CBB8476FF777

##### Dependency

##### Convert caffe model To Ascend om file

```bash
cd resnet
atc --model=./resnet50.prototxt --weight=./resnet50.caffemodel --framework=0 --output=./resnet50_aipp --soc_version=Ascend310 --insert_op_conf=./aipp_resnet50.cfg
```

##### Convert single op model To Ascend om file

```bash
cd ..
atc --singleop single_op/op_list.json --output ./single_op --soc_version=Ascend310
```


##### Versions that have been verified: 

- Atlas 800 (Model 3000)
- Atlas 300 (Model 3010)