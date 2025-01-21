#include<cuda.h>
#include<cuda_runtime_api.h>
#include<stdlib.h>
#include<opencv2/opencv.hpp>
#include<device_launch_parameters.h>

#include<NvInfer.h>
#include<iostream>
#include<fstream>
#include<string>
#include <vector>


using namespace nvinfer1;
using namespace std;


class Logger : public ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
}gLogger;


vector<string> load_class(string txt_file){
    std::ifstream file(txt_file); // 打开文件
    if (!file.is_open()) {
        std::cerr << "无法打开文件:"<<txt_file << std::endl;
    }
 
    std::vector<std::string> lines;
    std::string line;
 
    // 逐行读取文件内容，并存储到vector中
    while (std::getline(file, line)) {
        lines.push_back(line);
    }
 
    file.close(); // 关闭文件
    return lines;

}


float __device__ normalized(float src, float mean, float std)
{
    float res = (src / 255.0f - mean) / std;
    return res;
}

void __global__ preProcess(uchar * src, float* dst, const int width, const int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int size = width * height;
    // 使用imagenet中使用的均值方差
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std[3] = {0.229f, 0.224f, 0.225f};
    if (x < height && y < width)
    {
        int index = x + y * width;
        int b_index = index*3;
        int g_index = index*3+1;
        int r_index = index*3+2;

        // 进行归一化处理
        float b_normalized = normalized(src[b_index], mean[2], std[2]);
        float g_normalized = normalized(src[g_index], mean[1], std[1]);
        float r_normalized = normalized(src[r_index], mean[0], std[0]);
        // 将数据从HWC转化为CHW
        dst[index] = r_normalized;
        dst[index+size] = g_normalized;
        dst[index+size*2] = b_normalized;
    }

}


int main()
{
    const int width = 256, height=256;
    const int label_classes = 1000;

    //读取保存的二进制模型
    const std::string trtFile{ "./model.plan" };
    std::ifstream engineFile(trtFile, std::ios::binary);
    long int      fsize = 0;
    engineFile.seekg(0, engineFile.end);
    fsize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);
    std::vector<char> engineString(fsize);
    engineFile.read(engineString.data(), fsize);

    // 判断是否读取成果
    if (engineString.size() == 0)
    {
        std::cout << "Failed getting serialized engine!" << std::endl;
        return -1;
    }
    std::cout << "Succeeded getting serialized engine!" << std::endl;

    // 基于读取的文件构建推理引擎
    ICudaEngine* engine = nullptr;
    IRuntime* runtime{ createInferRuntime(gLogger) };
    engine = runtime->deserializeCudaEngine(engineString.data(), fsize);
    
    // 判断推理引擎是否构建成果
    if (engine == nullptr)
    {
        std::cout << "Failed loading engine!" << std::endl;
        return -2;
    }
    std::cout << "Succeeded loading engine!" << std::endl;


    // // 获取模型的输入、输出数量和对应名字
    // long unsigned int        nIO = engine->getNbIOTensors();
    // long unsigned int        nInput = 0;
    // long unsigned int        nOutput = 0;
    // std::vector<std::string> vTensorName(nIO);
    // for (int i = 0; i < nIO; ++i)
    // {
    //     vTensorName[i] = std::string(engine->getIOTensorName(i));
    //     nInput += int(engine->getTensorIOMode(vTensorName[i].c_str()) == TensorIOMode::kINPUT);
    //     nOutput += int(engine->getTensorIOMode(vTensorName[i].c_str()) == TensorIOMode::kOUTPUT);
    // }


    // 构建执行推理过程的context
    IExecutionContext* context = engine->createExecutionContext();

    // 设置执行推理的输入数据形状
    // context->setInputShape(vTensorName[0].c_str(), Dims64{ 4, {1, 3, height, width} });
    context->setInputShape("image", Dims64{ 4, {1, 3, height, width} });


    // 绑定模型的输入输出
    float* image, *label;
    cudaMalloc((void**)&image, height * width * 3 * sizeof(float));
    cudaMalloc((void**)&label, label_classes * sizeof(float));
    context->setTensorAddress("image", image);
    context->setTensorAddress("label", label);

    // 读取图片进行预处理
    std::string imgPath = "./test.jpg";
    cv::Mat img = cv::imread(imgPath);
    cv::resize(img, img, cv::Size(width, height));
    uchar *src;
    cudaMalloc((void**)&src, height*width*3*sizeof(uchar));
    cudaMemcpy(src, img.data, height*width*3*sizeof(uchar), cudaMemcpyHostToDevice);
    dim3 block_size(32, 32);
    dim3 grid_size((width + 32 - 1) / 32, (height + 32 - 1) / 32);
   
    preProcess << <grid_size, block_size >> > (src, image, width, height);


    // 进行推理
    context->enqueueV3(0);

    float* res_h = new float[label_classes];
    cudaMemcpy(res_h, label, label_classes * sizeof(float), cudaMemcpyDeviceToHost);

    // 读取imagenet数据集类别
    float max_prob = 0.0;
    int max_index = -1;
    vector<string> classes=load_class("imagenet_classes.txt");

    for (int i = 0; i < 1000; i++)
    {
        if (res_h[i] > max_prob)
        {
            max_prob = res_h[i];
            max_index = i;
        }
    }
    cout << "index:" << max_index << "\tprob:" <<max_prob <<"\tclass:"<<classes[max_index]<< endl;

    cudaFree(src);
    cudaFree(image);
    cudaFree(label);

    return 0;
}