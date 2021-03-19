EN|[CN](README(Linux).zh.md)
# InferClassification

## Introduction

This sample demonstrates how the InferClassification program to use the chip to perform the 'resnet50' target classification.
Process Framework

```
ReadJpeg > JpegDecode > ImageResize > ObjectClassification > Cast_Op > ArgMax_Op > WriteResult
```
## Supported Products

Atlas 800 (Model 3000), Atlas 800 (Model 3010), Atlas 300 (Model 3010), Atlas 500 (Model 3010), Atlas 300I (Model 6000)

## Supported ACL Version

1.73.5.1.B050, 1.73.5.2.B050, 1.75.T11.0.B116, 20.1.0, 20.2.0

Run the following command to check the version in the environment where the Atlas product is installed:
```bash
npu-smi info
```

## Dependency

The input model is Resnet50. Please refer to [model transformation instructions](data/models/README.md) to transform the model.

The single op is Cast and ArgMax. Please refer to [model transformation instructions](data/models/README.md) to transform the single op model.

Code dependency:

Each sample in the version package depends on the ascendbase directory.

If the whole package is not copied, ensure that the ascendbase and InferClassification directories are copied to the same directory in the compilation environment. Otherwise, the compilation will fail. If the whole package is copied, ignore it.

Set the environment variable:
*  `ASCEND_HOME`      Ascend installation path, which is generally `/usr/local/Ascend`
*  `LD_LIBRARY_PATH`  Specifies the dynamic library search path on which the sample program depends

```bash
export ASCEND_HOME=/usr/local/Ascend
export LD_LIBRARY_PATH=$ASCEND_HOME/ascend-toolkit/latest/acllib/lib64:$LD_LIBRARY_PATH
```

## Configuration

Configure the device_id, model_path and single_op_model in `data/config/setup.config`

Configure device_id
```bash
#chip config
device_id = 0 #use the device to run the program
```

Configure model_path
```bash
#resnet model path
model_path = ./data/models/resnet/resnet50_aipp.om
```

Configure model input format
```bash
#resnet model input width and height
model_width = 224
model_height = 224
```
Configure single op model path
```bash
#single op model path
single_op_model = ./data/models/single_op
```

## Compilation

Compile Atlas 800 (Model 3000), Atlas 800 (Model 3010), Atlas 300 (Model 3010) programs
```bash
bash build.sh
```

Compile Atlas 500 (Model 3010) program
Run the following command on the ARM server
```bash
bash build.sh
```

Run the following command on the X86 server
```bash
bash build.sh A500
```

If you want to run with the compilation result on another environment, copy the dist directory

## Execution

```bash
cd dist
./main -h

------------------------------help information------------------------------
-h                            help                          show helps
-help                         help                          show helps
-i                            ./data/test.jpg              Optional. Specify the input image, default: ./data/test.jpg

```

Classify the Jpeg images
```bash
cd dist
./main -i ./data/images/x4096.jpg
```

## Constraint
Support input format: Jpeg

Resolutions: maximum 4096 x 4096, minimum 32 x 32.

## Result

The classification result is generated in the out folder result/result.txt.

```bash
inference output index: 248
classname:  248: 'Eskimo dog, husky'
```
