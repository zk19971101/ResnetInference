import time
import cv2 as cv
import numpy as np
from cuda import cudart
import tensorrt as trt
from PIL import Image
from torchvision import transforms
from serialize_onnx import config

with open("./src/imagenet_classes.txt", 'r') as f:
    index_list = f.readlines()


def normalization(img_path):
    img = Image.open(img_path)
    data_transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         # transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img_ = data_transform(img)
    return img_


def normalization_v2(img_path):
    data = cv.imread(img_path)
    data = cv.cvtColor(data, cv.COLOR_BGR2RGB)
    data = cv.resize(data, (args.img_h, args.img_w))
    # 将图像从 [0, 255] 缩放到 [0.0, 1.0]
    img = data.astype(np.float32) / 255.0

    # 创建一个用于存储标准化后的图像的数组
    normalized_img = np.zeros_like(img, dtype=np.float32)
    # 每个通道的均值和标准差（例如，ImageNet数据集的常用值）
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    # 对每个通道进行标准化
    for c in range(3):
        normalized_img[:, :, c] = (img[:, :, c] - mean[c]) / std[c]
    normalized_img = normalized_img.transpose([2, 0, 1])
    return normalized_img


def load_plan(args):
    # logger记录运行过程中的信息
    logger = trt.Logger(trt.Logger.VERBOSE)
    # 读取序列化文件
    with open(args.trtFile, "rb") as f:
        engineString = f.read()

    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
    nIO = engine.num_io_tensors
    lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
    nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

    context = engine.create_execution_context()
    context.set_input_shape(lTensorName[0], [1, args.C, args.img_h, args.img_w])
    for i in range(nIO):
        print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]),
              engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

    bufferH = []
    inferenceImage = "./src/test.jpg"
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # normalized_img = normalization(inferenceImage).reshape(1, 3, 256, 256)
    normalized_img = normalization_v2(inferenceImage).reshape(3, 256, 256)

    bufferH.append(np.ascontiguousarray(normalized_img))
    for i in range(nInput, nIO):
        # a = context.get_tensor_shape(lTensorName[i])
        bufferH.append(np.empty(context.get_tensor_shape(lTensorName[i]),
                                dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))
    bufferD = []
    for i in range(nIO):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    for i in range(nInput):
        cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes,
                          cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    for i in range(nIO):
        context.set_tensor_address(lTensorName[i], int(bufferD[i]))

    start = time.time()
    context.execute_async_v3(0)
    duration = time.time() - start
    for i in range(nInput, nIO):
        cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes,
                          cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    for i in range(nInput, nIO):
        print(lTensorName[i])
        print(bufferH[i].argmax(-1))
        print(bufferH[i].max(-1))
        print(index_list[bufferH[i].argmax(-1)[0]])

    for b in bufferD:
        cudart.cudaFree(b)
    print(f"inference time is {duration} s")
    print("Succeeded running model in TensorRT!")


if __name__ == '__main__':
    args = config()
    load_plan(args)