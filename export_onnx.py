import torch
from torchvision.models import resnet34, ResNet34_Weights


def export_onnx(model, onnxFile):
    '''
    模型导出在GPU上进行
    :param model:
    :param onnxFile:
    :return:
    '''
    torch.onnx.export(
                model,
                torch.randn(1, 3, 256, 256, device='cuda'),
                onnxFile,
                # 输入、输出名，可以指定多个
                input_names=['image'],
                output_names=['label'],
                # 模型中是否包含权重信息
                do_constant_folding=True,
                verbose=True,
                keep_initializers_as_inputs=True,
                # 指定onnx算子集版本
                opset_version=12,
                # 指定输入输出张量动态维度，0表示batch_size为动态， 'inBatchSize'指定动态维度名字
                dynamic_axes={
                    'image': {0: 'inBatchSize'},
                    'label': {0: 'outBatchSize'}
                })
    print('onnx file has exported!')


if __name__ == '__main__':
    onnxFile = "./asset/resnet.onnx"
    weight = ResNet34_Weights.DEFAULT
    model = resnet34(weights=weight).to('cuda')
    export_onnx(model, onnxFile)