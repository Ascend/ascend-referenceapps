# mxIndex-test

#### 介绍
**本仓库提供了昇腾MindX SDK index组件实现的几种常见检索算法的demo**


#### 关于MindX SDK 更多信息
请关注昇腾社区[MindX SDK](https://www.hiascend.com/zh/software/mindx-sdk)的最新版本


#### 安装教程

1.  MindX SDK index [安装文档](https://www.hiascend.com/document/detail/zh/mind-sdk/300/featureretrieval/mxindexfrug/mxindexfrug_0001.html)

#### 测试用例说明
| 用例名称 | 用例说明 |
| ---------- | ------------------------------------------- |
|    TestAscendIndexFalt                 |   FP32转FP16 暴搜demo                                          |
|    TestAscendIndexInt8Falt             |   底库数据为int8 暴搜demo                                          |
|    TestAscendIndexInt8FaltWithSQ       |   FP32 SQ 量化为 int8 后, 暴搜demo                                  |
|    TestAscendIndexSQ                   |   FP32 SQ 量化为Int8后，反量化暴搜demo                              |
|    TestAscendIndexBinaryFlat           |   二值化底库特征汉明距离暴搜demo                              |
|    TestAscendIndexTS                   |   时空库，带属性过滤demo                                          |
|    TestAscendIndexTSV2                 |   时空库临时版本，正式mxIndex版本中，请使用TestAscendIndexTS，并删除TestAscendIndexTSV2)                                          |

#### Demo使用说明

1.  请先正确安装MindSDK Index组件及其依赖的driver、firmware、Ascend toolkit、OpenBLAS、Faiss
2.  执行一下命令编译demo
``` shell
bash build.sh
 ```
       
3.  在build目录中找到对应的二进制可执行文件
