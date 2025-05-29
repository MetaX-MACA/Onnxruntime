import onnx
import sys
import values, operation
import numpy as np
import log

from onnx import numpy_helper, helper

logger = log.getLogger(__name__, log.INFO)


#################################Pattern ONE
#MatMul->Reshape->BN->Reshape
def get_matmul_bn(model):
    mrbr_list= []

    for node in model.graph.node:
        if node.op_type == 'MatMul':
            reshape_node1, ok = operation.get_next_node_by_output(model, node.output[0])
            if ok == 0 and reshape_node1.op_type == 'Reshape':
                logger.debug('got reshape_node1: {}'.format(reshape_node1.name))
                bn_node, ok = operation.get_next_node_by_output(model, reshape_node1.output[0])
                if ok == 0:
                    if bn_node.op_type == 'BatchNormalization':
                        reshape_node2, ok = operation.get_next_node_by_output(model, bn_node.output[0])
                        if ok == 0:
                            if reshape_node2.op_type == 'Reshape':
                                logger.debug('----got match ConvTranspose+Add+BN node: {}'.format(node.name))

                                node_dict = {}
                                node_dict['MatMul'] = node
                                node_dict['Reshape1'] = reshape_node1
                                node_dict['BN'] = bn_node
                                node_dict['Reshape2'] = reshape_node2

                                mrbr_list.append(node_dict)

                                #for k, v in node_dict.items():
                                #    print('----name: {}, node: {}'.format(k, v.name))

    return mrbr_list      

def handle_crbr(model, mrbr_dict):
    mm_node = mrbr_dict['MatMul'] 
    bn_node = mrbr_dict['BN']
    reshape_node1 = mrbr_dict['Reshape1']
    reshape_node2 = mrbr_dict['Reshape2']

    matmul_b_name = mm_node.input[1]
    matmul_b_input = next(initializer for initializer in model.graph.initializer if initializer.name == matmul_b_name)
    matmul_b = numpy_helper.to_array(matmul_b_input)

    #data_type_str = onnx.TensorProto.DataType.Name(conv_weights_input.data_type)
    #print('data_type_str:', data_type_str)

    bn_scale_name = bn_node.input[1]
    bn_bias_name = bn_node.input[2]
    bn_mean_name = bn_node.input[3]
    bn_var_name = bn_node.input[4]

    bn_scale_input = next(initializer for initializer in model.graph.initializer if initializer.name == bn_scale_name)
    bn_bias_input = next(initializer for initializer in model.graph.initializer if initializer.name == bn_bias_name)
    bn_mean_input = next(initializer for initializer in model.graph.initializer if initializer.name == bn_mean_name)
    bn_var_input = next(initializer for initializer in model.graph.initializer if initializer.name == bn_var_name)

    bn_scale = numpy_helper.to_array(bn_scale_input)
    bn_bias = numpy_helper.to_array(bn_bias_input)
    bn_mean = numpy_helper.to_array(bn_mean_input)
    bn_var = numpy_helper.to_array(bn_var_input)

    bn_scale = bn_scale.reshape(1, bn_scale.shape[0])
    bn_var = bn_var.reshape(1, bn_var.shape[0])
    bn_mean = bn_mean.reshape(1, bn_mean.shape[0])
    bn_bias = bn_bias.reshape(1, bn_bias.shape[0])

    #print('+++ bn_scale:', bn_scale.shape)

    fused_gemm_c = bn_bias -  bn_mean * bn_scale / np.sqrt(bn_var)
    fused_gemm_c = fused_gemm_c.squeeze()
    #print('fused_gemm_c.shape:', fused_gemm_c.shape)

    fused_type = onnx.TensorProto.FLOAT
    if fused_gemm_c.dtype == np.float16:
        fused_type = onnx.TensorProto.FLOAT16

    gemm_c_name = bn_node.output[0] + '_to_gemm_c_'
    gemm_c = helper.make_tensor(gemm_c_name, fused_type, fused_gemm_c.shape, fused_gemm_c.flatten())
    model.graph.initializer.extend([gemm_c])

    fused_gemm_b =  matmul_b * bn_scale / np.sqrt(bn_var)
    #print('fused_gemm_b.shape:', fused_gemm_b.shape)

    gemm_b_name = bn_node.output[0] + '_to_gemm_b_'
    gemm_b = helper.make_tensor(gemm_b_name, fused_type, fused_gemm_b.shape, fused_gemm_b.flatten())
    model.graph.initializer.extend([gemm_b])

    reshape_node2.op_type = 'Gemm'
    reshape_node2.name = mm_node.name + '+' + bn_node.name + '+' + reshape_node1.name + '+' + reshape_node2.name
    reshape_node2.input[0] = mm_node.input[0]
    reshape_node2.input[1] = gemm_b_name
    reshape_node2.input.append(gemm_c_name)

    attr = onnx.helper.make_attribute('transA', 0)
    reshape_node2.attribute.append(attr)

    attr = onnx.helper.make_attribute('transB', 0)
    reshape_node2.attribute.append(attr)

    attr = onnx.helper.make_attribute('alpha', 1.0)
    reshape_node2.attribute.append(attr) 

    attr = onnx.helper.make_attribute('beta', 1.0)
    reshape_node2.attribute.append(attr)

    '''
    model.graph.initializer.remove(matmul_b_input)
    model.graph.initializer.remove(bn_scale_input)
    model.graph.initializer.remove(bn_bias_input)
    model.graph.initializer.remove(bn_mean_input)
    model.graph.initializer.remove(bn_var_input)

    reshape1_input1 = next(initializer for initializer in model.graph.initializer if initializer.name == reshape_node1.input[1])
    reshape2_input1 = next(initializer for initializer in model.graph.initializer if initializer.name == reshape_node1.input[1])

    model.graph.initializer.remove(reshape1_input1)
    model.graph.initializer.remove(reshape2_input1)
    '''

    operation.remove_onnx_node(model, mm_node)
    operation.remove_onnx_node(model, reshape_node1)
    operation.remove_onnx_node(model, bn_node)

def matmul_bn_to_gemm(model):
    mrbr_list = get_matmul_bn(model)

    for mrbr_dict in mrbr_list:
        handle_crbr(model, mrbr_dict)  

    return model

'''
if __name__ == "__main__":
    #model = onnx.load('/home/zqiu/models/det_mv3_db.onnx')
    model = onnx.load('/home/zqiu/models/facenet_sim.onnx')
    matmul_bn_to_gemm(model)
    onnx.save(model, './fs.onnx')
'''
    