import tensorrt as trt
import argparse


def run(args):
    # logger记录运行过程中的信息
    logger = trt.Logger(trt.Logger.VERBOSE)

    # builder作为模型搭建的入口，用于产生推理的engine和对网络进行配置、优化
    builder = trt.Builder(logger)

    # profile用于对网络进行优化，ie 进行模型的dynamic设置
    profile = builder.create_optimization_profile()

    # config对模型参数进行设置，比如设置模型使用显存、推理精度
    config = builder.create_builder_config()
    # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    # config.set_flag(trt.BuilderFlag.TF32)

    # 通过parser读取ONNX文件来构建网络主体
    # 设置网络为explicit batch模式，应对模型中对batch进行处理的层，比如Layer Normalization等
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    # 读取.onnx文件
    parser = trt.OnnxParser(network, logger)
    with open(args.onnxFile, "rb") as f:
        parser.parse(f.read())
        # if not parser.parse(f.read()):
        #     print("Failed parsing .onnx file!")
        #     for error in range(parser.num_errors):
        #         print(parser.get_error(error))
        #     exit()
        # print("Succeeded parsing .onnx file!")
    # 指定读取模型的输入、输出shape
    inputTensor = network.get_input(0)
    profile.set_shape(inputTensor.name, [args.min_batch, args.C, args.img_h, args.img_w],
                      [args.opt_batch, args.C, args.img_h, args.img_w],
                      [args.max_batch, args.C, args.img_h, args.img_w])
    config.add_optimization_profile(profile)
    # 将输出的第一个值遮住了了，导致输出的两个值变为一个
    # network.unmark_output(network.get_output(0))

    # 将模型转化为trt支持的二进制中间形态，方便后续快速读取调用
    # 存在将网络序列化时网络为None
    engine_string = builder.build_serialized_network(network, config)
    print("engine", type(engine_string))
    with open(args.trtFile, "wb") as f:
        f.write(engine_string)


def config():
    parser = argparse.ArgumentParser(description="TRT inference")

    parser.add_argument('--img_h', default=256)
    parser.add_argument('--img_w', default=256)
    parser.add_argument('--min_batch', default=1)
    parser.add_argument('--opt_batch', default=4)
    parser.add_argument('--max_batch', default=8)
    parser.add_argument('--C', default=3)

    # parser.add_argument('--onnxFile', default='./asset/resnet.onnx')
    parser.add_argument('--onnxFile', default='./asset/resnet.onnx')
    # parser.add_argument('--trtFile', default='./asset/model.plan')
    parser.add_argument('--trtFile', default='./asset/resnet34_bs_dynamic_1-4-8.engine')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = config()
    run(args)
