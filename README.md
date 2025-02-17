# C++通过TensorRT实现resnet推理加速

## 模型导出为onnx
在export_onnx.py通过torch.onnx.export（）将模型转化为onnx文件。
## 转化为engine
- 方法一：
通过代码进行转换
在serialize_onnx.py实现
- 方法二：通过命令行进行转换

>固定batchsize
> trtexec --onnx=resnet.onnx --saveEngine=resnet32_bs_1.engine

>动态batchsize使用
>trtexec --onnx=resnet.onnx --saveEngine=resnet34_bs_dynamic_1-4-8.engine --timingCacheFile=dynamic-1-4-8.cache --minShapes=image:1x3x256x256 --maxShapes=image:8x3x256x256 --optShapes=image:4x3x256x256

>设置shape时要使用x而不是*

## 通过opencv读取图像并通过cuda对数据进行前处理
**由于通过opencv读取到的图像为BGR格式，通过指针进行逐像素索引的顺序是BGRBGRBGR，在tensorRT中传入的像素顺序由神经网络模型推理的通道顺序确定的，在resnet中tensor为[C,H,W]的顺序传入的。**


## 通过TensorRT进行推理加速
1. 读取保存的engine文件构建engine
2. 构建执行推理过程的context，通过context设置执行推理的输入数据形状、占用显存大小
3. 在显存中开辟空间，绑定模型的输入输出指针
4. 对图像进行前处理，进行模型推理，进行后处理输出结果

## 测试结果
输入图片
![test](https://github.com/user-attachments/assets/38d3e339-0907-4f1c-afad-d463ed2edf70)

推理结果
![图片](https://github.com/user-attachments/assets/0e77f45e-3bd0-4fbc-8b20-f349f15b06ae)
