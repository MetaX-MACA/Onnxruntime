import onnx
import correct_batch
import numpy as np
import values
import log
import operation

from onnx import numpy_helper, helper

logger = log.getLogger(__name__, log.INFO)

def matmul_add_to_gemm(model):
    dict_matmul = {}
    dict_add = {}

    got_matmul_add = False

    search = True

    pads = []

    index = 0

    while search == True:
        search = False

        for node_id, node in enumerate(model.graph.node):
            #print(node_id, ", name:", node.name, ", input:", node.input, ", output:", node.output,  \
            #         ", op:", node.op_type, ', len(input):', len(node.input))
            if node.op_type == 'MatMul':
                input0_shape = values.get_tensor_shape_by_name(model, node.input[0])
                if len(input0_shape) == 0:
                    _, input0_shape = values.get_init_value_and_shape(model, node.input[0])

                input1_shape = values.get_tensor_shape_by_name(model, node.input[1])
                if len(input1_shape) == 0:
                    _, input1_shape = values.get_init_value_and_shape(model, node.input[1])

                if len(input0_shape) == 2 and len(input1_shape) == 2:
                    dict_matmul['input'] = node.input
                    dict_matmul['output'] = node.output
                    dict_matmul['id'] = node_id

                    #print('got match matmul: {}'.format(node.name))

                    next_node, _ = operation.get_next_node_by_output(model, node.output[0])
                    if next_node.op_type == 'Add':
                        if len(dict_matmul) > 0 and dict_matmul['output'][0] in next_node.input:
                            dict_add['input'] = next_node.input
                            dict_add['output'] = next_node.output
                            #dict_add['id'] = node_id
                            logger.debug('got matmul+add pair, matmul: {} {}'.format(dict_matmul['input'], dict_matmul['output']))
                            logger.debug('got matmul+add pair, add: {} {}'.format(dict_add['input'], dict_add['output']))

                            got_matmul_add = True

                            c = next_node.input[1]
                            if dict_matmul['output'][0] == next_node.input[1]:
                                c = next_node.input[0]

                            next_node.op_type = 'Gemm'
                            next_node.name = node.name + '+' + next_node.name
                            next_node.input[0] = dict_matmul['input'][0]
                            next_node.input[1] = dict_matmul['input'][1]
                            next_node.input.append(c)

                            attr = onnx.helper.make_attribute('transA', 0)
                            next_node.attribute.append(attr)

                            attr = onnx.helper.make_attribute('transB', 0)
                            next_node.attribute.append(attr)

                            attr = onnx.helper.make_attribute('alpha', 1.0)
                            next_node.attribute.append(attr) 

                            attr = onnx.helper.make_attribute('beta', 1.0)
                            next_node.attribute.append(attr)   

                            old_node = model.graph.node[dict_matmul['id']] 
                            model.graph.node.remove(old_node)

                            dict_matmul = {}
                            dict_add = {}
                            search = True
                            break
                        else:
                            logger.debug('clear matmul+add dict')
                            dict_matmul = {}
                            dict_add = {}
                elif len(input0_shape) > 2 and len(input1_shape) == 2:
                    logger.debug('------input0_shape:{}, input1_shape:{}'.format(input0_shape, input1_shape))
                    '''
                    dict_matmul['input'] = node.input
                    dict_matmul['output'] = node.output
                    dict_matmul['id'] = node_id

                    next_node, _ = operation.get_next_node_by_output(model, node.output[0])
                    if next_node.op_type == 'Add':
                        if len(dict_matmul) > 0 and dict_matmul['output'][0] in next_node.input:
                            dict_add['input'] = next_node.input
                            dict_add['output'] = next_node.output
                            #dict_add['id'] = node_id
                            logger.debug('got matmul+add pair, matmul: {} {}'.format(dict_matmul['input'], dict_matmul['output']))
                            logger.debug('got matmul+add pair, add: {} {}'.format(dict_add['input'], dict_add['output']))

                            got_matmul_add = True

                            c = next_node.input[1]
                            if dict_matmul['output'][0] == next_node.input[1]:
                                c = next_node.input[0]

                            shape0 = 1
                            for i in range(len(input0_shape) -1):
                                shape0 = shape0 * input0_shape[i]

                            if shape0 <= 0:
                                shape0 = -1

                            new_shape = [shape0, input0_shape[-1]]
                            #print('matmul input0 new_shape:', new_shape)

                            node.op_type = 'Reshape'

                            const_shape_name = node.input[0] + '_reshape_data_' + str(index)
                            
                            const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                                                data_type=onnx.TensorProto.INT64,
                                                dims=[len(new_shape)],
                                                vals=new_shape)

                            model.graph.initializer.append(const_shape_tensor)

                            MatMul_B = node.input[1]

                            node.input[1] = const_shape_name

                            operation.update_tensor_shape(model, node.output[0], new_shape)

                            next_node.op_type = 'Gemm'
                            next_node.input[0] = node.output[0]
                            next_node.input[1] = MatMul_B
                            next_node.input.append(c)


                            old_shape = values.get_tensor_shape_by_name(model, next_node.output[0])
                            if len(old_shape) > 0:
                                shape0 = 1
                                for i in range(len(old_shape) -1):
                                    shape0 = shape0 * old_shape[i]

                                if shape0 <= 0:
                                    shape0 = -1

                                new_shape = [shape0, old_shape[-1]]
                                #print('gemm output new_shape:', new_shape)

                            operation.update_tensor_shape(model, next_node.output[0], new_shape)

                            attr = onnx.helper.make_attribute('transA', 0)
                            next_node.attribute.append(attr)

                            attr = onnx.helper.make_attribute('transB', 0)
                            next_node.attribute.append(attr)

                            attr = onnx.helper.make_attribute('alpha', 1.0)
                            next_node.attribute.append(attr) 

                            attr = onnx.helper.make_attribute('beta', 1.0)
                            next_node.attribute.append(attr)

                            nnext_node, _ = operation.get_next_node_by_output(model, next_node.output[0])
                            if nnext_node.op_type == 'Reshape':
                                reshape_input1_name = nnext_node.input[1]
                                reshape_input1_input = next(initializer for initializer in model.graph.initializer if initializer.name == reshape_input1_name)
                                reshape_input1 = numpy_helper.to_array(reshape_input1_input)
                                if reshape_input1[0] <= 0:
                                    print('XXXXXXXXXXX reshape_input1:', reshape_input1_name, reshape_input1)
                                    input_shape = values.get_tensor_shape_by_name(model, nnext_node.input[0])
                                    if len(input_shape) > 0:
                                        print('YYYYYY input_shape[0]:', input_shape[0]) 
                                        output_shape = values.get_tensor_shape_by_name(model, nnext_node.output[0])
                                        if len(output_shape) > 0:
                                            print('ZZZZZZ output_shape[0]:', output_shape[0])
                                            if input_shape[0] != output_shape[0]:
                                                logger.warning('Need to modify reshape data(input_shape[0]({}) != output_shape[0]({}))'.format(input_shape[0], output_shape[0]))
                                                reshape_input1_list = reshape_input1.tolist()
                                                reshape_input1_list[0] = output_shape[0]

                                                fused_type = onnx.TensorProto.INT64
                                                #if reshape_input1.dtype == np.float16:
                                                #    fused_type = onnx.TensorProto.FLOAT16

                                                new_reshape_input1_name = reshape_input1_name+'_new_'
                                                new_reshape_input1 = helper.make_tensor(new_reshape_input1_name, fused_type, reshape_input1.shape, reshape_input1_list)#reshape_input1.flatten())
                                                model.graph.initializer.extend([new_reshape_input1])
                                                nnext_node.input[1] = new_reshape_input1_name

                                                operation.remove_initializer_if_necessary_by_name(model, reshape_input1_name, nnext_node)
                            else:
                                #print('Need to insert Reshape node before:', nnext_node.name)
                                ################
                                if len(old_shape) > 0:
                                    shape_tensor_name = next_node.output[0] + '_reshape_data_' + str(index)
                                    const_shape = onnx.helper.make_tensor(shape_tensor_name, onnx.TensorProto.INT64, [len(old_shape)], old_shape)
                                    model.graph.initializer.append(const_shape)

                                    output_tensor_name = next_node.output[0] + '_reshape_' + str(index)
                                    output_tensor = onnx.helper.make_tensor_value_info(output_tensor_name, onnx.TensorProto.FLOAT, old_shape)

                                    model.graph.value_info.append(output_tensor)

                                    rs_node = onnx.helper.make_node(
                                        name=next_node.output[0]+'__Reshape_Node_' + str(index),
                                        op_type='Reshape', 
                                        inputs=[next_node.output[0], shape_tensor_name],
                                        outputs=[output_tensor_name]
                                        )

                                    all_next_nodes, _ = operation.get_all_next_node_by_output(model, next_node.output[0])
                                    for nn in all_next_nodes:
                                        for i, input_ in enumerate(nn.input):
                                            if next_node.output[0] == input_:
                                                nn.input[i] = output_tensor_name

                                    operation.insert_onnx_node(model, rs_node, next_node)
                                ##################    

                            index = index + 1 
                            
                            dict_matmul = {}
                            dict_add = {}
                            search = True
                            break
                        else:
                            logger.debug('clear matmul+add dict')
                            dict_matmul = {}
                            dict_add = {}
                    '''
    if got_matmul_add == True:
        logger.debug('got matmul+add node------------')

    return model

