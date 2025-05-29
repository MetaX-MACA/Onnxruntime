import onnx
import sys
import values, operation
import numpy as np
import log
from onnx import numpy_helper, helper

logger = log.getLogger(__name__, log.INFO)

transpose_node_map = {}
reshape_node_map = {}

def get_prev_node_by_input(model, input_):
    n = model.graph.node[0]
    for node in model.graph.node:
        if input_ in node.output:
            return node, 0

    return n, -1

def get_next_node_by_output(model, output):
    n = model.graph.node[0]
    for node in model.graph.node:
        if output in node.input:
            return node, 0

    return n, -1

def get_all_next_node_by_output(model, output):
    node_list = []
    ok = -1

    for node in model.graph.node:
        if output in node.input:
            node_list.append(node)
            ok = 0

    return node_list, ok

def insert_node(model, insert_node, follow_up_node):
    # 根据插入Node的输出修改后续node的输入
    #follow_up_node.input[0] = insert_node.output[0]
    # 找到后续Node的索引位置，并将插入节点插入到graph中
    for follow_up_node_index, _follow_up_node in enumerate(model.graph.node):
        if _follow_up_node == follow_up_node:
            logger.debug("follow_up_node_index: {}".format(follow_up_node_index))
            model.graph.node.insert(follow_up_node_index, insert_node)
            break

def get_node_group(model, input_name, num, index):
    node_list = []
    name = input_name
    for i in range(num):
        node, ok = get_prev_node_by_input(model, name)
        if ok == 0 and len(node.input) > index[i]:
            name = node.input[index[i]]
            node_list.append(node)
        else:
            break

    return node_list

##################
def get_unuse_node_list(model):
    unuse_node_list = []

    for node in model.graph.node:
        if node.op_type == 'Shape':
            gather_node, ok = operation.get_next_node_by_output(model, node.output[0])
            if ok == 0 and gather_node.op_type == 'Gather':
                all_next_node, ok = operation.get_all_next_node_by_output(model, gather_node.output[0])
                for mul_node in all_next_node:
                    if ok == 0 and mul_node.op_type == 'Mul':
                        unsqueeze_node, ok = operation.get_next_node_by_output(model, mul_node.output[0])
                        if ok == 0 and unsqueeze_node.op_type == 'Unsqueeze':
                            concat_node, ok = operation.get_next_node_by_output(model, unsqueeze_node.output[0])
                            if ok == 0 and concat_node.op_type == 'Concat':
                                _, ok = operation.get_next_node_by_output(model, concat_node.output[0])
                                if ok == -1:
                                    #print('got match concat node:', concat_node.name)
                                    d = {}
                                    d['Concat'] = concat_node
                                    d['Unsqueeze'] = unsqueeze_node
                                    d['Mul'] = mul_node
                                    unuse_node_list.append(d)

    return unuse_node_list

def handle_unused_node_list(model):
    unuse_node_list = get_unuse_node_list(model)
    for d in unuse_node_list:
        concat_node = d['Concat']
        us_node = d['Unsqueeze']
        mul_node = d['Mul']

        operation.remove_onnx_node(model, concat_node)
        operation.remove_onnx_node(model, us_node)
        operation.remove_onnx_node(model, mul_node)

def del_unused_concat(model):
    for node in model.graph.node:
        if node.op_type == 'Concat':
            _, ok = operation.get_next_node_by_output(model, node.output[0])
            if ok == -1:
                #print('delete unused concat node:', node.name)
                operation.remove_onnx_node(model, node)

def del_unused_unsqueeze(model):
    for node in model.graph.node:
        if node.op_type == 'Unsqueeze':
            _, ok = operation.get_next_node_by_output(model, node.output[0])
            if ok == -1:
                #print('delete unused Unsqueeze node:', node.name)
                operation.remove_onnx_node(model, node)

def get_unuse_node_list2(model):
    unuse_node_list = []

    for node in model.graph.node:
        if node.op_type == 'Shape':
            gather_node, ok = operation.get_next_node_by_output(model, node.output[0])
            if ok == 0 and gather_node.op_type == 'Gather':
                unsqueeze_node, ok = operation.get_next_node_by_output(model, gather_node.output[0])
                if ok == 0 and unsqueeze_node.op_type == 'Unsqueeze':
                    concat_node, ok = operation.get_next_node_by_output(model, unsqueeze_node.output[0])
                    if ok == 0 and concat_node.op_type == 'Concat':
                        _, ok = operation.get_next_node_by_output(model, concat_node.output[0])
                        if ok == -1:
                            #print('got match concat node:', concat_node.name)
                            d = {}
                            d['Shape'] = node
                            d['Gather'] = gather_node
                            d['Unsqueeze'] = unsqueeze_node
                            d['Concat'] = concat_node
                            unuse_node_list.append(d)

    return unuse_node_list

def handle_unused_node_list2(model):
    unuse_node_list = get_unuse_node_list2(model)
    for d in unuse_node_list:
        shape_node = d['Shape']
        gather_node = d['Gather']
        us_node = d['Unsqueeze']
        concat_node = d['Concat']

        operation.remove_onnx_node(model, concat_node)
        operation.remove_onnx_node(model, us_node)
        operation.remove_onnx_node(model, gather_node)
        operation.remove_onnx_node(model, shape_node)

def get_unuse_node_list3(model):
    unuse_node_list = []

    for node in model.graph.node:
        if node.op_type == 'Expand':
            _, ok = get_next_node_by_output(model, node.output[0])
            if ok == -1:
                where_node, ok = operation.get_prev_node_by_input(model, node.input[1])
                #print('get_unuse_node_list3, where_node.op_type=', where_node.op_type)
                if ok == 0 and where_node.op_type == 'Where':
                    equal_node, _ = get_prev_node_by_input(model, where_node.input[0])
                    concat_node, _ = get_prev_node_by_input(model, equal_node.input[0])
                    us_node1, _ = get_prev_node_by_input(model, concat_node.input[0])
                    gather_node1, _ = get_prev_node_by_input(model, us_node1.input[0])
                    shape_node, ok = get_prev_node_by_input(model, gather_node1.input[0])
                    if ok == 0 and shape_node.op_type == 'Shape':
                        us_node2, _ = get_prev_node_by_input(model, node.input[0])
                        where_node2, _ = get_prev_node_by_input(model, us_node2.input[0])
                        less_node, _ = get_prev_node_by_input(model, where_node2.input[0])
                        reshape_node, _ = get_prev_node_by_input(model, less_node.input[1])
                        add_node, _ = get_prev_node_by_input(model, reshape_node.input[0])
                        range_node, _ = get_prev_node_by_input(model, add_node.input[0])
                        squeeze_node, _ = get_prev_node_by_input(model, range_node.input[1])
                        slice_node, _ = get_prev_node_by_input(model, squeeze_node.input[0])
                        shape_node2, _ = get_prev_node_by_input(model, slice_node.input[0])
                        cos_node, _ = get_prev_node_by_input(model, shape_node2.input[0])
                        concat_node2, _ = get_prev_node_by_input(model, cos_node.input[0])
                        us_node3, _ = get_prev_node_by_input(model, concat_node2.input[0])
                        gather_node2, _ = get_prev_node_by_input(model, us_node3.input[0])
                        shape_node3, ok = get_prev_node_by_input(model, gather_node2.input[0])
                        #print('get_unuse_node_list3, shape_node3.name:{}, shape_node.name:{}, ok:{}'.format(shape_node3.name, shape_node.name, ok))
                        if shape_node3 == shape_node:
                            print('got match expand node:', node.name)
                            d = {}
                            d['Expand'] = node
                            d['Where1'] = where_node
                            d['Equal'] = equal_node
                            d['Concat1'] = concat_node
                            d['Us1'] = us_node1
                            d['Gather1'] = gather_node1

                            d['Shape'] = shape_node
                            d['Us2'] = us_node2
                            d['Where2'] = where_node2
                            d['Less'] = less_node
                            d['Reshape'] = reshape_node
                            d['Add'] = add_node
                            d['Range'] = range_node
                            d['Squeeze'] = squeeze_node
                            d['Slice'] = slice_node
                            d['Shape2'] = shape_node2
                            d['COS'] = cos_node
                            d['Concat2'] = concat_node2
                            d['Us2'] = us_node2
                            d['Gather2'] = gather_node2

                            unuse_node_list.append(d)

    return unuse_node_list

def handle_unused_node_list3(model):
    unuse_node_list = get_unuse_node_list3(model)
    for d in unuse_node_list:
        expand_node = d['Expand']
        where_node = d['Where1']
        equal_node = d['Equal']
        concat_node = d['Concat1']
        us_node1 = d['Us1']
        gather_node1 = d['Gather1']

        shape_node = d['Shape']
        us_node2 = d['Us2']
        where_node2 = d['Where2']
        less_node = d['Less']
        reshape_node = d['Reshape']
        add_node = d['Add']
        range_node = d['Range']
        squeeze_node = d['Squeeze']
        shape_node2 = d['Shape2']
        cos_node = d['COS']
        concat_node2 = d['Concat2']
        us_node2 = d['Us2']
        gather_node2 = d['Gather2']
        slice_node = d['Slice']

        operation.remove_onnx_node(model, expand_node)
        operation.remove_onnx_node(model, where_node)
        operation.remove_onnx_node(model, equal_node)
        operation.remove_onnx_node(model, concat_node)
        operation.remove_onnx_node(model, us_node1)
        operation.remove_onnx_node(model, gather_node1)


        operation.remove_onnx_node(model, concat_node2)
        operation.remove_onnx_node(model, cos_node)
        operation.remove_onnx_node(model, shape_node2)
        operation.remove_onnx_node(model, slice_node)
        operation.remove_onnx_node(model, squeeze_node)
        operation.remove_onnx_node(model, range_node)
        operation.remove_onnx_node(model, add_node)
        operation.remove_onnx_node(model, reshape_node)
        operation.remove_onnx_node(model, less_node)
        operation.remove_onnx_node(model, where_node2)
        operation.remove_onnx_node(model, us_node2)
##############################


#################################Pattern ONE
#Transpose-->MatMul-->Div-->Softmax-->MatMul-->Tranpose
def get_tmdsms(model):
    res = -1

    node_list= []

    for node in model.graph.node:
        skip = False
        if node.op_type == 'Transpose':
            input_info = next((x for x in model.graph.value_info if x.name == node.input[0]), None)

            if input_info:
                input_dtype = input_info.type.tensor_type.elem_type

                if input_dtype == onnx.TensorProto.FLOAT16:
                    #print(f"Input '{node.input[0]}' of operator '{node.name}' is of type FP16.")
                    logger.debug("Input {} of operator {} is of type FP16.".format(node.input[0], node.name))
                else:
                    #print(f"Input '{node.input[0]}' of operator '{node.name}' is not of type FP16.")
                    continue
            else:
                #print(f"Could not find information for input '{node.input[0]}' of operator '{node.name}'.")
                continue

            if len(node_list) > 0:
                for d in node_list:
                    for _, v in d.items():
                        if v.name == node.name:
                            logger.debug('node: {} has been processed'.format(node.name))
                            skip = True
                            break

            if skip == True:
                logger.debug('node: {} has been processed, skip it'.format(node.name))
                continue

            mm_node1, ok = get_next_node_by_output(model, node.output[0])
            if ok == 0 and mm_node1.op_type == 'MatMul':
                logger.debug('got matmul_node: {}'.format(mm_node1.name))
                div_node, ok = get_next_node_by_output(model, mm_node1.output[0])
                if ok == 0 and div_node.op_type == 'Div':
                    softmax_node, ok = get_next_node_by_output(model, div_node.output[0])
                    if ok == 0 and softmax_node.op_type == 'Softmax':
                        mm_node2, ok = get_next_node_by_output(model, softmax_node.output[0])
                        if ok == 0 and mm_node2.op_type == 'MatMul':
                            tp_node, ok = get_next_node_by_output(model, mm_node2.output[0])
                            if ok == 0 and tp_node.op_type == 'Transpose':
                                mm1_inputB = node
                                mm1_another_input = mm_node1.input[1]
                                if mm_node1.input[1] == node.output[0]:
                                    mm1_another_input = mm_node1.input[0]

                                mm1_another_input_node, _ = get_prev_node_by_input(model, mm1_another_input)
                                if mm1_another_input_node.op_type == 'Transpose':
                                    mm1_inputA = mm1_another_input_node

                                    if mm_node1.input[0] == node.output[0]:
                                        mm1_inputA = node
                                        mm1_inputB = mm1_another_input_node

                                    mm2_another_input = mm_node2.input[1]
                                    if mm_node2.input[1] == softmax_node.output[0]:
                                        mm2_another_input = mm_node2.input[0]

                                    mm2_another_input_node, _ = get_prev_node_by_input(model, mm2_another_input)
                                    if mm2_another_input_node.op_type == 'Transpose':
                                        logger.debug('----got match Transpose node: {}'.format(node.name))

                                        res = 0

                                        node_dict = {}
                                        node_dict['tp1'] = mm1_inputA
                                        node_dict['tp2'] = mm1_inputB
                                        node_dict['mm1'] = mm_node1
                                        node_dict['div'] = div_node
                                        node_dict['softmax'] = softmax_node
                                        node_dict['mm2'] = mm_node2
                                        node_dict['tp'] = tp_node
                                        node_dict['tp3'] = mm2_another_input_node

                                        node_list.append(node_dict)

                                        #for k, v in node_dict.items():
                                        #    print('+++++ name: {}, node: {}'.format(k, v.name))

    return node_list, res

def match_mha_block_pattern_one(model):
    res = -1

    for node in model.graph.node:
        if node.op_type == 'Transpose':
            node_list = get_node_group(model, node.input[0], 5, [0,0,0,0,0])
            if len(node_list) == 5:
                #expected_pattern = ['MatMul', 'Div', 'Softmax', 'MatMul', 'Transpose']
                expected_pattern = ['MatMul', 'Softmax', 'Div', 'MatMul', 'Transpose']
                for idx1, n in enumerate(node_list):
                    #print('node:', idx1, n.op_type, expected_pattern[idx1])
                    if n.op_type != expected_pattern[idx1]:
                        break

                if idx1 == 4:
                    res = 0
                    #print('match_mha_block_pattern_one, got it!!!!')
                    break
    return res

#Transpose-->MatMul-->Div-->Softmax-->MatMul-->Tranpose
def handle_mha_block_pattern_one(model):
    node_list, ok = get_tmdsms(model)
    if ok == 0:
        index = 0
        for node_dict in node_list:
            matmul_node1 = node_dict['mm1']
            matmul_node2 = node_dict['mm2']
            tp_node1 = node_dict['tp1']
            tp_node2 = node_dict['tp2']
            tp_node3 = node_dict['tp3']
            tp_node = node_dict['tp']
            div_node = node_dict['div']
            softmax_node = node_dict['softmax']

            logger.debug('start handle tmdsms, matmul name:{}'.format(matmul_node1.name))

            mha_node_name = 'MultiHeadAttention_pattern1_' + str(index)
            index = index + 1

            mha_output_name = mha_node_name + '_output_'
            q_output_shape = values.get_tensor_shape_by_name(model, tp_node1.input[0])
            k_output_shape = values.get_tensor_shape_by_name(model, tp_node2.input[0])
            v_output_shape = values.get_tensor_shape_by_name(model, tp_node2.input[0])

            #####
            head_dim_ = -1
            num_heads_ = -1
            num_kv_heads_ = -1

            if len(q_output_shape) >= 2:
                head_dim_ = q_output_shape[-1]
                num_heads_ = q_output_shape[-2]

            if len(k_output_shape) >= 2:
                num_kv_heads_ = k_output_shape[-2]
            ####

            if head_dim_ > 256:
                return

            logger.debug('----q_shape:{}, k_shape:{}, v_shape:{}'.format(q_output_shape, k_output_shape, v_output_shape))
            mha_output_shape = q_output_shape
            mha_output = onnx.helper.make_tensor_value_info(mha_output_name, onnx.TensorProto.FLOAT16, mha_output_shape)

            mha_node = onnx.helper.make_node(
                                                'MultiHeadAttentionV1',
                                                name=mha_node_name,
                                                inputs=[tp_node1.input[0], tp_node2.input[0], tp_node3.input[0]],
                                                outputs=[mha_output_name],
                                                head_dim=head_dim_, #q_output_shape[-1],
                                                is_causal=1,
                                                num_heads=num_heads_, #q_output_shape[-2],
                                                num_kv_heads=num_kv_heads_, #k_output_shape[-2],
                                                domain='com.metax-tech')

            model.graph.value_info.append(mha_output)

            insert_node(model, mha_node, tp_node)

            next_node, _ = get_next_node_by_output(model, tp_node.output[0])
            for i, input_ in enumerate(next_node.input):
                if tp_node.output[0] == input_:
                    next_node.input[i] = mha_output_name

            tp3_all_next_nodes, _ = operation.get_all_next_node_by_output(model, tp_node3.output[0])
            #print('len(all_next_nodes):', len(tp3_all_next_nodes))

            operation.remove_onnx_node(model, matmul_node1)
            operation.remove_onnx_node(model, matmul_node2)
            operation.remove_onnx_node(model, tp_node1)
            operation.remove_onnx_node(model, tp_node2)

            if len(tp3_all_next_nodes) == 1:
                operation.remove_onnx_node(model, tp_node3)

            operation.remove_onnx_node(model, tp_node)
            operation.remove_onnx_node(model, div_node)
            operation.remove_onnx_node(model, softmax_node)

#################################Pattern TWO
def match_mha_block_pattern_two(model):
    res = -1

    for node in model.graph.node:
        if node.op_type == 'Transpose':
            node_list = get_node_group(model, node.input[0], 6, [0,0,0,0,0,0])
            if len(node_list) == 6:
                #expected_pattern = ['MatMul', 'Div', 'Softmax', 'MatMul', 'Transpose']
                expected_pattern = ['MatMul', 'Softmax', 'Add', 'Div', 'MatMul', 'Transpose']
                for idx1, n in enumerate(node_list):
                    #print('node:', idx1, n.op_type, expected_pattern[idx1])
                    if n.op_type != expected_pattern[idx1]:
                        break

                if idx1 == 5:
                    res = 0
                    #print('match_mha_block_pattern_two, got it!!!!')
                    break
    return res

def match_mha_block_pattern_two_ext(model):
    res = -1

    for node in model.graph.node:
        if node.op_type == 'Transpose':
            node_list = get_node_group(model, node.input[0], 6, [0,0,0,0,0,0])
            if len(node_list) == 6:
                #expected_pattern = ['MatMul', 'Div', 'Softmax', 'MatMul', 'Transpose']
                expected_pattern = ['MatMul', 'Softmax', 'Add', 'Mul', 'MatMul', 'Transpose']
                for idx1, n in enumerate(node_list):
                    #print('node:', idx1, n.op_type, expected_pattern[idx1])
                    if n.op_type != expected_pattern[idx1]:
                        break

                if idx1 == 5:
                    res = 0
                    #print('match_mha_block_pattern_two, got it!!!!')
                    break
    return res

#Transpose-->MatMul-->Div or Mul-->Add-->Softmax-->MatMul-->Tranpose
def get_tmdasms(model):
    res = -1

    node_list= []

    for node in model.graph.node:
        skip = False
        if node.op_type == 'Transpose':
            input_info = next((x for x in model.graph.value_info if x.name == node.input[0]), None)

            if input_info:
                input_dtype = input_info.type.tensor_type.elem_type

                if input_dtype == onnx.TensorProto.FLOAT16:
                    #print(f"Input '{node.input[0]}' of operator '{node.name}' is of type FP16.")
                    logger.debug("Input {} of operator {} is of type FP16.".format(node.input[0], node.name))
                else:
                    #print(f"Input '{node.input[0]}' of operator '{node.name}' is not of type FP16.")
                    continue
            else:
                #print(f"Could not find information for input '{node.input[0]}' of operator '{node.name}'.")
                continue

            if len(node_list) > 0:
                for d in node_list:
                    for _, v in d.items():
                        if v.name == node.name:
                            logger.debug('node: {} has been processed'.format(node.name))
                            skip = True
                            break

            if skip == True:
                logger.debug('node: {} has been processed, skip it'.format(node.name))
                continue

            mm_node1, ok = get_next_node_by_output(model, node.output[0])
            if ok == 0 and mm_node1.op_type == 'MatMul':
                logger.debug('got matmul_node: {}'.format(mm_node1.name))
                div_node, ok = get_next_node_by_output(model, mm_node1.output[0])
                if ok == 0 and (div_node.op_type == 'Div' or div_node.op_type == 'Mul'):
                #if ok == 0 and div_node.op_type == 'Div':
                    add_node, ok = get_next_node_by_output(model, div_node.output[0])
                    if ok == 0 and add_node.op_type == 'Add':
                        softmax_node, ok = get_next_node_by_output(model, add_node.output[0])
                        if ok == 0 and softmax_node.op_type == 'Softmax':
                            mm_node2, ok = get_next_node_by_output(model, softmax_node.output[0])
                            if ok == 0 and mm_node2.op_type == 'MatMul':
                                tp_node, ok = get_next_node_by_output(model, mm_node2.output[0])
                                if ok == 0 and tp_node.op_type == 'Transpose':
                                    mm1_inputB = node
                                    mm1_another_input = mm_node1.input[1]
                                    if mm_node1.input[1] == node.output[0]:
                                        mm1_another_input = mm_node1.input[0]

                                    mm1_another_input_node, _ = get_prev_node_by_input(model, mm1_another_input)
                                    if mm1_another_input_node.op_type == 'Transpose':
                                        mm1_inputA = mm1_another_input_node

                                        if mm_node1.input[0] == node.output[0]:
                                            mm1_inputA = node
                                            mm1_inputB = mm1_another_input_node

                                        mm2_another_input = mm_node2.input[1]
                                        if mm_node2.input[1] == softmax_node.output[0]:
                                            mm2_another_input = mm_node2.input[0]

                                        mm2_another_input_node, _ = get_prev_node_by_input(model, mm2_another_input)
                                        if mm2_another_input_node.op_type == 'Transpose':
                                            logger.debug('----got match Transpose node: {}'.format(node.name))

                                            res = 0

                                            node_dict = {}
                                            node_dict['tp1'] = mm1_inputA
                                            node_dict['tp2'] = mm1_inputB
                                            node_dict['mm1'] = mm_node1
                                            node_dict['div'] = div_node
                                            node_dict['softmax'] = softmax_node
                                            node_dict['mm2'] = mm_node2
                                            node_dict['tp'] = tp_node
                                            node_dict['tp3'] = mm2_another_input_node
                                            node_dict['add'] = add_node

                                            node_list.append(node_dict)

                                            #for k, v in node_dict.items():
                                            #    print('---name: {}, node: {}'.format(k, v.name))

    return node_list, res


#Transpose-->MatMul-->Div or Mul-->Add-->Softmax-->MatMul-->Tranpose
def handle_mha_block_pattern_two(model):
    transpose_map = {}
    node_list, ok = get_tmdasms(model)
    if ok == 0:
        index = 0
        for node_dict in node_list:
            matmul_node1 = node_dict['mm1']
            matmul_node2 = node_dict['mm2']
            tp_node1 = node_dict['tp1']
            tp_node2 = node_dict['tp2']
            tp_node3 = node_dict['tp3']
            tp_node = node_dict['tp']
            div_node = node_dict['div']
            add_node = node_dict['add']
            softmax_node = node_dict['softmax']

            logger.debug('start handle tmdasms, matmul name:{}'.format(matmul_node1.name))

            mha_node_name = 'MultiHeadAttention_pattern2_' + str(index)
            index = index + 1

            mha_output_name = mha_node_name + '_output_'
            q_output_shape = values.get_tensor_shape_by_name(model, tp_node1.input[0])
            k_output_shape = values.get_tensor_shape_by_name(model, tp_node2.input[0])
            #v_output_shape = values.get_tensor_shape_by_name(model, tp_node2.input[0])

            #logger.debug('q_shape:{}, k_shape:{}, v_shape:{}'.format(q_output_shape, k_output_shape, v_output_shape))

            head_dim_ = -1
            num_heads_ = -1
            num_kv_heads_ = -1

            if len(q_output_shape) >= 2:
                head_dim_ = q_output_shape[-1]
                num_heads_ = q_output_shape[-2]

            if len(k_output_shape) >= 2:
                num_kv_heads_ = k_output_shape[-2]

            if head_dim_ > 256:
                return

            input3 = add_node.input[1]
            if add_node.input[1] == div_node.output[0]:
                input3 = add_node.input[0]

            mha_node = onnx.helper.make_node(
                                                'MultiHeadAttentionV1',
                                                name=mha_node_name,
                                                #inputs=[tp_node1.input[0], tp_node2.input[0], tp_node3.input[0]],
                                                inputs=[tp_node1.input[0], tp_node2.input[0], tp_node3.input[0], input3],
                                                outputs=[mha_output_name],
                                                head_dim=head_dim_,
                                                is_causal=0,
                                                num_heads=num_heads_,
                                                num_kv_heads=num_kv_heads_,
                                                domain='com.metax-tech')

            mha_output_shape = q_output_shape
            if len(mha_output_shape):
                #print('got mha_output_shape:', mha_output_shape)
                mha_output = onnx.helper.make_tensor_value_info(mha_output_name, onnx.TensorProto.FLOAT16, mha_output_shape)
                model.graph.value_info.append(mha_output)

            insert_node(model, mha_node, tp_node)

            all_next_node, _ = get_all_next_node_by_output(model, tp_node.output[0])
            for next_node in all_next_node:
                for i, input_ in enumerate(next_node.input):
                    if tp_node.output[0] == input_:
                        next_node.input[i] = mha_output_name


            tp3_all_next_nodes, _ = operation.get_all_next_node_by_output(model, tp_node3.output[0])
            #print('len(all_next_nodes):', len(tp3_all_next_nodes))

            operation.remove_onnx_node(model, matmul_node1)
            operation.remove_onnx_node(model, matmul_node2)
            operation.remove_onnx_node(model, tp_node1)
            operation.remove_onnx_node(model, tp_node2)

            if len(tp3_all_next_nodes) == 1:
                operation.remove_onnx_node(model, tp_node3)

            operation.remove_onnx_node(model, tp_node)
            operation.remove_onnx_node(model, div_node)
            operation.remove_onnx_node(model, add_node)
            operation.remove_onnx_node(model, softmax_node)


################ pattern THREE
def match_mha_block_pattern_three(model):
    res = -1

    for node in model.graph.node:
        if node.op_type == 'Transpose':
            #print('node.name:', node.name)
            node_list = get_node_group(model, node.input[0], 6, [0,0,0,0,0,0])
            if len(node_list) == 6:
                expected_pattern = ['Reshape', 'MatMul', 'Softmax', 'Mul', 'MatMul', 'Transpose']
                for idx1, n in enumerate(node_list):
                    #print('-- node:', idx1, n.op_type, expected_pattern[idx1])
                    if n.op_type != expected_pattern[idx1]:
                        #print('xxxx  node:', idx1, n.op_type, expected_pattern[idx1])
                        break

                if idx1 == 5:
                    res = 0
                    print('match_mha_block_pattern_three, got it!!!!')
                    break
    return res

#Transpose-->MatMul-->Mul-->Softmax-->MatMul-->Reshape
def get_tmmsmr(model):
    res = -1

    node_list= []

    for node in model.graph.node:
        skip = False
        if node.op_type == 'Transpose':
            input_info = next((x for x in model.graph.value_info if x.name == node.input[0]), None)

            if input_info:
                input_dtype = input_info.type.tensor_type.elem_type

                if input_dtype == onnx.TensorProto.FLOAT16:
                    #print(f"Input '{node.input[0]}' of operator '{node.name}' is of type FP16.")
                    logger.debug("Input {} of operator {} is of type FP16.".format(node.input[0], node.name))
                else:
                    #print(f"Input '{node.input[0]}' of operator '{node.name}' is not of type FP16.")
                    continue
            else:
                #print(f"Could not find information for input '{node.input[0]}' of operator '{node.name}'.")
                continue

            if len(node_list) > 0:
                for d in node_list:
                    for _, v in d.items():
                        if v.name == node.name:
                            logger.debug('node: {} has been processed'.format(node.name))
                            skip = True
                            break

            if skip == True:
                logger.debug('node: {} has been processed, skip it'.format(node.name))
                continue

            mm_node1, ok = get_next_node_by_output(model, node.output[0])
            if ok == 0 and mm_node1.op_type == 'MatMul':
                logger.debug('got matmul_node: {}'.format(mm_node1.name))
                mul_node, ok = get_next_node_by_output(model, mm_node1.output[0])
                if ok == 0 and mul_node.op_type == 'Mul':
                    softmax_node, ok = get_next_node_by_output(model, mul_node.output[0])
                    if ok == 0 and softmax_node.op_type == 'Softmax':
                        mm_node2, ok = get_next_node_by_output(model, softmax_node.output[0])
                        if ok == 0 and mm_node2.op_type == 'MatMul':
                            reshape_node1, ok = get_next_node_by_output(model, mm_node2.output[0])
                            if ok == 0 and reshape_node1.op_type == 'Reshape':
                                mm1_inputB = node
                                mm1_another_input = mm_node1.input[1]
                                if mm_node1.input[1] == node.output[0]:
                                    mm1_another_input = mm_node1.input[0]

                                mm1_another_input_node, _ = get_prev_node_by_input(model, mm1_another_input)
                                if mm1_another_input_node.op_type == 'Reshape':
                                    mm1_inputA = mm1_another_input_node

                                    if mm_node1.input[0] == node.output[0]:
                                        mm1_inputA = node
                                        mm1_inputB = mm1_another_input_node

                                    mm2_another_input = mm_node2.input[1]
                                    if mm_node2.input[1] == softmax_node.output[0]:
                                        mm2_another_input = mm_node2.input[0]

                                    mm2_another_input_node, _ = get_prev_node_by_input(model, mm2_another_input)
                                    if mm2_another_input_node.op_type == 'Reshape':
                                        logger.debug('----got match Transpose node: {}'.format(node.name))

                                        res = 0

                                        node_dict = {}
                                        node_dict['reshape0'] = mm1_inputA
                                        node_dict['tp'] = mm1_inputB
                                        node_dict['mm1'] = mm_node1
                                        node_dict['mul'] = mul_node
                                        node_dict['softmax'] = softmax_node
                                        node_dict['mm2'] = mm_node2
                                        node_dict['reshape1'] = mm2_another_input_node
                                        node_dict['reshape2'] = reshape_node1

                                        node_list.append(node_dict)

                                        '''
                                        print('reshape0:', mm1_inputA.name)
                                        print('tp:', mm1_inputB.name)
                                        print('mm_node1:', mm_node1.name)
                                        print('mul:', mul_node.name)
                                        print('softmax_node:', softmax_node.name)
                                        print('mm_node2:', mm_node2.name)
                                        print('reshape1:', mm2_another_input_node.name)
                                        print('reshape2:', reshape_node1.name)
                                        '''

                                        #for k, v in node_dict.items():
                                        #    print('name: {}, node: {}'.format(k, v.name))

    return node_list, res

def handle_mha_block_pattern_three(model):
    node_list, ok = get_tmmsmr(model)
    if ok == 0:
        index = 0
        for node_dict in node_list:
            matmul_node1 = node_dict['mm1']
            matmul_node2 = node_dict['mm2']
            tp_node = node_dict['tp']
            rs_node0 = node_dict['reshape0']
            rs_node1 = node_dict['reshape1']
            rs_node2 = node_dict['reshape2']
            mul_node = node_dict['mul']
            softmax_node = node_dict['softmax']

            rs_node0_prev, _ = operation.get_prev_node_by_input(model, rs_node0.input[0])
            rs_node1_prev, _ = operation.get_prev_node_by_input(model, rs_node1.input[0])

            tp_node_prev, _ = operation.get_prev_node_by_input(model, tp_node.input[0])
            tp_node_prev_prev, _ = operation.get_prev_node_by_input(model, tp_node_prev.input[0])

            rs_node2_next, _ = operation.get_next_node_by_output(model, rs_node2.output[0])

            logger.debug('start handle tmmsmr, matmul name:{}'.format(matmul_node1.name))

            mha_node_name = 'MultiHeadAttention_pattern3_' + str(index)
            index = index + 1

            mha_output_name = mha_node_name + '_output_'
            q_output_shape = values.get_tensor_shape_by_name(model, rs_node0_prev.input[0])
            k_output_shape = q_output_shape #values.get_tensor_shape_by_name(model, tp_node2.input[0])
            v_output_shape = q_output_shape #values.get_tensor_shape_by_name(model, tp_node2.input[0])

            #####
            head_dim_ = -1
            num_heads_ = -1
            num_kv_heads_ = -1

            if len(q_output_shape) >= 2:
                head_dim_ = q_output_shape[-1]
                num_heads_ = q_output_shape[-2]

            if len(k_output_shape) >= 2:
                num_kv_heads_ = k_output_shape[-2]
            ####
            if head_dim_ > 256:
                return

            logger.debug('==== q_shape:{}, k_shape:{}, v_shape:{}'.format(q_output_shape, k_output_shape, v_output_shape))

            mha_output_shape = q_output_shape
            mha_output = onnx.helper.make_tensor_value_info(mha_output_name, onnx.TensorProto.FLOAT16, mha_output_shape)


            mha_node = onnx.helper.make_node(
                                                'MultiHeadAttentionV1',
                                                name=mha_node_name,
                                                inputs=[rs_node0_prev.input[0], tp_node_prev_prev.input[0], rs_node1_prev.input[0]],
                                                outputs=[mha_output_name],
                                                head_dim=head_dim_, #q_output_shape[-1],
                                                is_causal=0,
                                                num_heads=num_heads_, #q_output_shape[-2],
                                                num_kv_heads=num_kv_heads_, #k_output_shape[-2],
                                                domain='com.metax-tech')

            model.graph.value_info.append(mha_output)

            insert_node(model, mha_node, tp_node)

            next_node, _ = get_next_node_by_output(model, rs_node2_next.output[0])
            for i, input_ in enumerate(next_node.input):
                if rs_node2_next.output[0] == input_:
                    next_node.input[i] = mha_output_name

            operation.remove_onnx_node(model, matmul_node1)
            operation.remove_onnx_node(model, matmul_node2)
            operation.remove_onnx_node(model, tp_node)
            operation.remove_onnx_node(model, rs_node0)
            operation.remove_onnx_node(model, rs_node0_prev)
            operation.remove_onnx_node(model, rs_node1)
            operation.remove_onnx_node(model, rs_node1_prev)
            operation.remove_onnx_node(model, rs_node2)
            operation.remove_onnx_node(model, rs_node2_next)
            operation.remove_onnx_node(model, mul_node)
            operation.remove_onnx_node(model, softmax_node)
            operation.remove_onnx_node(model, tp_node_prev)
            operation.remove_onnx_node(model, tp_node_prev_prev)

################ pattern FOUR
def match_mha_block_pattern_four(model):
    res = -1

    for node in model.graph.node:
        if node.op_type == 'Transpose':
            #print('node.name:', node.name)
            node_list = get_node_group(model, node.input[0], 6, [0,0,0,0,0,0])
            if len(node_list) == 6:
                expected_pattern = ['Reshape', 'MatMul', 'Softmax', 'Reshape', 'Add', 'Reshape']
                for idx1, n in enumerate(node_list):
                    #print('-- node:', idx1, n.op_type, expected_pattern[idx1])
                    if n.op_type != expected_pattern[idx1]:
                        #print('xxxx  node:', idx1, n.op_type, expected_pattern[idx1])
                        break

                if idx1 == 5:
                    res = 0
                    print('match_mha_block_pattern_four, got it!!!!')
                    break
    return res

################ pattern FIVE
def match_mha_block_pattern_five(model):
    res = -1

    for node in model.graph.node:
        if node.op_type == 'Transpose':
            #print('node.name:', node.name)
            node_list = get_node_group(model, node.input[0], 6, [0,0,0,0,0,0])
            if len(node_list) == 6:
                expected_pattern = ['MatMul', 'Softmax', 'MatMul', 'Mul/Div', 'Transpose', 'Reshape']
                for idx1, n in enumerate(node_list):
                    #print('-- node:', idx1, n.op_type, expected_pattern[idx1])
                    if n.op_type not in expected_pattern[idx1].split('/'):
                        #print('xxxx  node:', idx1, n.op_type, expected_pattern[idx1])
                        break

                if idx1 == 5:
                    res = 0
                    print('match_mha_block_pattern_five, got it!!!!')
                    break
    return res

################ pattern SIX
def match_mha_block_pattern_six(model):
    res = -1

    for node in model.graph.node:
        if node.op_type == 'Transpose':
            #print('node.name:', node.name)
            node_list = get_node_group(model, node.input[0], 7, [0,0,0,0,0,0,0])
            if len(node_list) == 7:
                expected_pattern = ['MatMul', 'Reshape', 'Softmax', 'Add', 'Div', 'Reshape', 'MatMul']
                for idx1, n in enumerate(node_list):
                    #print('-- node:', idx1, n.op_type, expected_pattern[idx1])
                    if n.op_type != expected_pattern[idx1]:
                        #print('xxxx  node:', idx1, n.op_type, expected_pattern[idx1])
                        break

                if idx1 == 6:
                    res = 0
                    print('match_mha_block_pattern_six, got it!!!!')
                    break
    return res

#Transpose-->MatMul-->Reshape-->Add-->Reshape-->Softmax-->MatMul
def get_tmrarsm(model):
    res = -1

    node_list= []

    for node in model.graph.node:
        skip = False
        if node.op_type == 'Transpose':
            input_info = next((x for x in model.graph.value_info if x.name == node.input[0]), None)

            if input_info:
                input_dtype = input_info.type.tensor_type.elem_type

                if input_dtype == onnx.TensorProto.FLOAT16:
                    #print(f"Input '{node.input[0]}' of operator '{node.name}' is of type FP16.")
                    logger.debug("Input {} of operator {} is of type FP16.".format(node.input[0], node.name))
                else:
                    #print(f"Input '{node.input[0]}' of operator '{node.name}' is not of type FP16.")
                    continue
            else:
                #print(f"Could not find information for input '{node.input[0]}' of operator '{node.name}'.")
                continue

            if len(node_list) > 0:
                for d in node_list:
                    for _, v in d.items():
                        if v.name == node.name:
                            logger.debug('node: {} has been processed'.format(node.name))
                            skip = True
                            break

            if skip == True:
                logger.debug('node: {} has been processed, skip it'.format(node.name))
                continue

            mm_node1, ok = get_next_node_by_output(model, node.output[0])
            if ok == 0 and mm_node1.op_type == 'MatMul':
                logger.debug('got matmul_node: {}'.format(mm_node1.name))
                rs_node1, ok = get_next_node_by_output(model, mm_node1.output[0])
                if ok == 0 and rs_node1.op_type == 'Reshape':
                    add_node, ok = get_next_node_by_output(model, rs_node1.output[0])
                    if ok == 0 and add_node.op_type == 'Add':
                        rs_node2, ok = get_next_node_by_output(model, add_node.output[0])
                        if ok == 0 and rs_node2.op_type == 'Reshape':
                            softmax_node, ok = get_next_node_by_output(model, rs_node2.output[0])
                            if ok == 0 and softmax_node.op_type == 'Softmax':
                                mm_node2, ok = get_next_node_by_output(model, softmax_node.output[0])
                                if ok == 0 and mm_node2.op_type == 'MatMul':
                                    rs_node3, ok = get_next_node_by_output(model, mm_node2.output[0])
                                    if ok == 0 and rs_node3.op_type == 'Reshape':
                                        mm1_inputB = node
                                        mm1_another_input = mm_node1.input[1]
                                        if mm_node1.input[1] == node.output[0]:
                                            mm1_another_input = mm_node1.input[0]

                                        mm1_another_input_node, _ = get_prev_node_by_input(model, mm1_another_input)
                                        if mm1_another_input_node.op_type == 'Reshape':
                                            mm1_inputA = mm1_another_input_node

                                            if mm_node1.input[0] == node.output[0]:
                                                mm1_inputA = node
                                                mm1_inputB = mm1_another_input_node

                                            mm2_another_input = mm_node2.input[1]
                                            if mm_node2.input[1] == softmax_node.output[0]:
                                                mm2_another_input = mm_node2.input[0]

                                            mm2_another_input_node, _ = get_prev_node_by_input(model, mm2_another_input)
                                            if mm2_another_input_node.op_type == 'Reshape':
                                                logger.debug('xxxx got match Transpose node: {}'.format(node.name))

                                                res = 0

                                                node_dict = {}
                                                node_dict['reshape0'] = mm1_inputA
                                                node_dict['reshape1'] = rs_node1
                                                node_dict['reshape2'] = rs_node2
                                                node_dict['tp'] = mm1_inputB
                                                node_dict['mm1'] = mm_node1
                                                node_dict['add'] = add_node
                                                node_dict['reshape1'] = rs_node1
                                                node_dict['softmax'] = softmax_node
                                                node_dict['mm2'] = mm_node2
                                                node_dict['reshape3'] = mm2_another_input_node
                                                node_dict['reshape4'] = rs_node3

                                                node_list.append(node_dict)

                                                '''
                                                print('reshape0:', mm1_inputA.name)
                                                print('tp:', mm1_inputB.name)
                                                print('mm_node1:', mm_node1.name)
                                                print('reshape1:', rs_node1.name)
                                                print('add:', add_node.name)
                                                print('reshape2:', rs_node2.name)
                                                print('softmax_node:', softmax_node.name)
                                                print('mm_node2:', mm_node2.name)
                                                print('reshape3:', mm2_another_input_node.name)
                                                print('reshape4:', rs_node3.name)
                                                '''

                                                #for k, v in node_dict.items():
                                                #    print('name: {}, node: {}'.format(k, v.name))

    return node_list, res

def handle_mha_block_pattern_four(model):
    node_list, ok = get_tmrarsm(model)
    if ok == 0:
        index = 0
        for node_dict in node_list:
            matmul_node1 = node_dict['mm1']
            matmul_node2 = node_dict['mm2']
            tp_node = node_dict['tp']
            rs_node0 = node_dict['reshape0']
            rs_node1 = node_dict['reshape1']
            rs_node2 = node_dict['reshape2']
            rs_node3 = node_dict['reshape3']
            rs_node4 = node_dict['reshape4']
            add_node = node_dict['add']
            softmax_node = node_dict['softmax']


            rs_node0_prev, _ = operation.get_prev_node_by_input(model, rs_node0.input[0])
            rs_node0_pprev, _ = operation.get_prev_node_by_input(model, rs_node0_prev.input[0])
            rs_node0_ppprev, _ = operation.get_prev_node_by_input(model, rs_node0_pprev.input[0])

            tp_node_prev, _ = operation.get_prev_node_by_input(model, tp_node.input[0])
            tp_node_prev_prev, _ = operation.get_prev_node_by_input(model, tp_node_prev.input[0])

            rs_node3_prev, _ = operation.get_prev_node_by_input(model, rs_node3.input[0])

            rs_node4_next, _ = operation.get_next_node_by_output(model, rs_node4.output[0])

            logger.debug('start handle tmmsmr, matmul name:{}'.format(matmul_node1.name))

            mha_node_name = 'MultiHeadAttention_pattern4_' + str(index)
            index = index + 1

            mha_output_name = mha_node_name + '_output_'
            q_output_shape = values.get_tensor_shape_by_name(model, rs_node0_prev.input[0])
            k_output_shape = q_output_shape #values.get_tensor_shape_by_name(model, tp_node2.input[0])
            v_output_shape = q_output_shape #values.get_tensor_shape_by_name(model, tp_node2.input[0])


            #####
            head_dim_ = -1
            num_heads_ = -1
            num_kv_heads_ = -1

            if len(q_output_shape) >= 2:
                head_dim_ = q_output_shape[-1]
                num_heads_ = q_output_shape[-2]

            if len(k_output_shape) >= 2:
                num_kv_heads_ = k_output_shape[-2]
            ####

            if head_dim_ > 256:
                return

            logger.debug('==== q_shape:{}, k_shape:{}, v_shape:{}'.format(q_output_shape, k_output_shape, v_output_shape))

            mha_output_shape = q_output_shape
            mha_output = onnx.helper.make_tensor_value_info(mha_output_name, onnx.TensorProto.FLOAT16, mha_output_shape)


            mha_node = onnx.helper.make_node(
                                                'MultiHeadAttentionV1',
                                                name=mha_node_name,
                                                inputs=[rs_node0_prev.input[0], tp_node_prev_prev.input[0], rs_node3_prev.input[0]],
                                                outputs=[mha_output_name],
                                                head_dim=head_dim_, #q_output_shape[-1],
                                                is_causal=1,
                                                num_heads=num_heads_, #q_output_shape[-2],
                                                num_kv_heads=num_kv_heads_, #k_output_shape[-2],
                                                domain='com.metax-tech')

            if q_output_shape != [0, 0, 0, 0]:
                model.graph.value_info.append(mha_output)

            insert_node(model, mha_node, tp_node)

            next_node, _ = get_next_node_by_output(model, rs_node4_next.output[0])
            for i, input_ in enumerate(next_node.input):
                if rs_node4_next.output[0] == input_:
                    next_node.input[i] = mha_output_name

            rs_node0_pprev.input[0] = rs_node0_ppprev.input[0]

            operation.remove_onnx_node(model, matmul_node1)
            operation.remove_onnx_node(model, matmul_node2)
            operation.remove_onnx_node(model, tp_node)
            operation.remove_onnx_node(model, rs_node0)
            operation.remove_onnx_node(model, rs_node1)
            operation.remove_onnx_node(model, rs_node2)
            operation.remove_onnx_node(model, rs_node3)
            operation.remove_onnx_node(model, rs_node4)
            operation.remove_onnx_node(model, add_node)
            operation.remove_onnx_node(model, softmax_node)
            operation.remove_onnx_node(model, tp_node_prev)
            operation.remove_onnx_node(model, rs_node0_prev)
            operation.remove_onnx_node(model, rs_node0_ppprev)
            operation.remove_onnx_node(model, tp_node_prev_prev)
            operation.remove_onnx_node(model, rs_node3_prev)
            operation.remove_onnx_node(model, rs_node4_next)

        handle_unused_node_list(model)
        del_unused_concat(model)
        handle_unused_node_list2(model)
        handle_unused_node_list3(model)

###
#Transpose-->Mul-->MatMul-->Softmax-->MatMul-->Transpose
def get_tmmsmt(model):
    res = -1

    node_list= []

    for node in model.graph.node:
        skip = False
        if node.op_type == 'Transpose':
            input_info = next((x for x in model.graph.value_info if x.name == node.input[0]), None)

            if input_info:
                input_dtype = input_info.type.tensor_type.elem_type

                if input_dtype == onnx.TensorProto.FLOAT16:
                    #print(f"Input '{node.input[0]}' of operator '{node.name}' is of type FP16.")
                    logger.debug("Input {} of operator {} is of type FP16.".format(node.input[0], node.name))
                else:
                    #print(f"Input '{node.input[0]}' of operator '{node.name}' is not of type FP16.")
                    continue
            else:
                #print(f"Could not find information for input '{node.input[0]}' of operator '{node.name}'.")
                continue

            if len(node_list) > 0:
                for d in node_list:
                    for _, v in d.items():
                        #print('---k:{}, v:{}'.format(k, v.name))
                        if v.name == node.name:
                            logger.debug('node: {} has been processed'.format(node.name))
                            #print('node: {} has been processed'.format(node.name))
                            skip = True
                            break

            if skip == True:
                logger.debug('node: {} has been processed, skip it'.format(node.name))
                continue

            mul_node1, ok = get_next_node_by_output(model, node.output[0])
            if ok == 0 and (mul_node1.op_type == 'Mul' or mul_node1.op_type == 'Div'):
                mm_node1, ok = get_next_node_by_output(model, mul_node1.output[0])
                if ok == 0 and mm_node1.op_type == 'MatMul':
                    logger.debug('got matmul_node: {}'.format(mm_node1.name))
                    softmax_node, ok = get_next_node_by_output(model, mm_node1.output[0])
                    if ok == 0 and softmax_node.op_type == 'Softmax':
                        mm_node2, ok = get_next_node_by_output(model, softmax_node.output[0])
                        if ok == 0 and mm_node2.op_type == 'MatMul':
                            ts_node2, ok = get_next_node_by_output(model, mm_node2.output[0])
                            if ok == 0 and ts_node2.op_type == 'Transpose':
                                mm1_inputB = mul_node1
                                mm1_another_input = mm_node1.input[0]
                                if mm_node1.input[0] == mul_node1.output[0]:
                                    mm1_another_input = mm_node1.input[1]

                                mm1_another_input_node, _ = get_prev_node_by_input(model, mm1_another_input)
                                next_node_op_type_list = ['Mul', 'Div', 'Transpose']
                                if mm1_another_input_node.op_type in next_node_op_type_list:
                                    mm1_inputA = mm1_another_input_node
                                    ts_node3, _ = get_prev_node_by_input(model, mm1_another_input_node.input[0])
                                    if mm1_another_input_node.op_type == 'Transpose':
                                        ts_node3 = mm1_another_input_node

                                    if mm_node1.input[0] == mul_node1.output[0]:
                                        mm1_inputA = mul_node1
                                        mm1_inputB = mm1_another_input_node

                                    mm2_another_input = mm_node2.input[1]
                                    if mm_node2.input[1] == softmax_node.output[0]:
                                        mm2_another_input = mm_node2.input[0]

                                    mm2_another_input_node, _ = get_prev_node_by_input(model, mm2_another_input)
                                    if mm2_another_input_node.op_type == 'Transpose':
                                        logger.debug('xxxx got match Transpose node: {}'.format(node.name))

                                        res = 0

                                        node_dict = {}
                                        #node_dict['tp'] = node
                                        node_dict['mm1'] = mm_node1
                                        node_dict['mm1_inputA'] = mm1_inputA
                                        node_dict['mm1_inputB'] = mm1_inputB
                                        node_dict['softmax'] = softmax_node
                                        node_dict['tp2'] = ts_node2
                                        #node_dict['tp3'] = ts_node3
                                        node_dict['mm2'] = mm_node2
                                        node_dict['mm2_inputB'] = mm2_another_input_node

                                        if mm1_inputA.input[0] == node.output[0]:
                                            node_dict['tp'] = node
                                            node_dict['tp3'] = ts_node3
                                        else:
                                            node_dict['tp'] = ts_node3
                                            node_dict['tp3'] = node

                                        node_list.append(node_dict)

                                        '''
                                        print('tp:', node.name)
                                        print('tp3:', ts_node3.name)
                                        print('tp2:', ts_node2.name)
                                        print('mm1:', mm_node1.name)
                                        print('mm1_inputA:', mm1_inputA.name)
                                        print('mm1_inputB:', mm1_inputB.name)
                                        print('softmax_node:', softmax_node.name)
                                        print('mm2:', mm_node2.name)
                                        '''


    return node_list, res

#Shape-->Slice-->Cast-->Sqrt-->Div-->Sqrt-->None
def get_sscsds(model):
    node_list = []
    for node in model.graph.node:
        if node.op_type == 'Shape':
            slice_node, ok = get_next_node_by_output(model, node.output[0])
            if ok == 0 and slice_node.op_type == 'Slice':
                cast_node, ok = get_next_node_by_output(model, slice_node.output[0])
                if ok == 0 and cast_node.op_type == 'Cast':
                    sqrt_node1, ok = get_next_node_by_output(model, cast_node.output[0])
                    if ok == 0 and sqrt_node1.op_type == 'Sqrt':
                        div_node, ok = get_next_node_by_output(model, sqrt_node1.output[0])
                        if ok == 0 and div_node.op_type == 'Div':
                            sqrt_node2, ok = get_next_node_by_output(model, div_node.output[0])
                            _, ok = get_next_node_by_output(model, sqrt_node2.output[0])
                            if ok == -1:
                                #print('got match shape node:', node.name)
                                d = {}
                                d['Shape'] = node
                                d['Slice'] = slice_node
                                d['Cast'] = cast_node
                                d['Sqrt1'] = sqrt_node1
                                d['Div'] = div_node
                                d['Sqrt2'] = sqrt_node2

                                node_list.append(d)

    return node_list

def handle_sscsds(model):
    node_list = get_sscsds(model)
    for d in node_list:
        shape_node = d['Shape']
        slice_node = d['Slice']
        cast_node = d['Cast']
        sqrt_node1 = d['Sqrt1']
        div_node = d['Div']
        sqrt_node2 = d['Sqrt2']

        operation.remove_onnx_node(model, sqrt_node2)
        operation.remove_onnx_node(model, div_node)
        operation.remove_onnx_node(model, sqrt_node1)
        operation.remove_onnx_node(model, cast_node)
        operation.remove_onnx_node(model, slice_node)
        operation.remove_onnx_node(model, shape_node)

def handle_mha_block_pattern_five(model):
    node_list, ok = get_tmmsmt(model)
    if ok == 0:
        index = 0
        for node_dict in node_list:
            matmul_node1 = node_dict['mm1']
            matmul_node2 = node_dict['mm2']
            tp_node = node_dict['tp']
            mm1_inputA = node_dict['mm1_inputA']
            mm1_inputB = node_dict['mm1_inputB']
            softmax_node = node_dict['softmax']
            tp_node = node_dict['tp']
            tp_node2 = node_dict['tp2']
            tp_node3 = node_dict['tp3']
            mm2_inputB = node_dict['mm2_inputB']


            logger.debug('start handle tmmsmr, matmul name:{}'.format(matmul_node1.name))

            mha_node_name = 'MultiHeadAttention_pattern5_' + str(index)
            index = index + 1

            mha_output_name = mha_node_name + '_output_'

            q_output_shape = values.get_tensor_shape_by_name(model, tp_node.input[0])
            k_output_shape = q_output_shape #values.get_tensor_shape_by_name(model, tp_node2.input[0])
            v_output_shape = q_output_shape #values.get_tensor_shape_by_name(model, tp_node2.input[0])

            #####
            head_dim_ = -1
            num_heads_ = -1
            num_kv_heads_ = -1

            if len(q_output_shape) >= 2:
                head_dim_ = q_output_shape[-1]
                num_heads_ = q_output_shape[-2]

            if len(k_output_shape) >= 2:
                num_kv_heads_ = k_output_shape[-2]
            ####

            if head_dim_ == 0:
                head_dim_ = -1

            if num_heads_ == 0:
                num_heads_ = -1

            if num_kv_heads_ == 0:
                num_kv_heads_ = -1

            if head_dim_ > 256:
                return

            logger.debug('==== q_shape:{}, k_shape:{}, v_shape:{}'.format(q_output_shape, k_output_shape, v_output_shape))

            mha_output_shape = q_output_shape
            mha_output = onnx.helper.make_tensor_value_info(mha_output_name, onnx.TensorProto.FLOAT16, mha_output_shape)


            mha_node = onnx.helper.make_node(
                                                'MultiHeadAttentionV1',
                                                name=mha_node_name,
                                                inputs=[tp_node.input[0], tp_node3.input[0], mm2_inputB.input[0]],
                                                outputs=[mha_output_name],
                                                head_dim=head_dim_, #q_output_shape[-1],
                                                is_causal=0,
                                                num_heads=num_heads_, #q_output_shape[-2],
                                                num_kv_heads=num_kv_heads_, #k_output_shape[-2],
                                                domain='com.metax-tech')

            model.graph.value_info.append(mha_output)

            insert_node(model, mha_node, tp_node)

            next_node, _ = get_next_node_by_output(model, tp_node2.output[0])
            for i, input_ in enumerate(next_node.input):
                if tp_node2.output[0] == input_:
                    next_node.input[i] = mha_output_name

            operation.remove_onnx_node(model, tp_node)
            operation.remove_onnx_node(model, tp_node3)
            operation.remove_onnx_node(model, mm1_inputA)
            operation.remove_onnx_node(model, mm1_inputB)
            operation.remove_onnx_node(model, matmul_node1)

            operation.remove_onnx_node(model, softmax_node)
            operation.remove_onnx_node(model, mm2_inputB)
            operation.remove_onnx_node(model, matmul_node2)

            operation.remove_onnx_node(model, tp_node2)

        handle_sscsds(model)

###
#Transpose-->MatMul-->Reshape-->Div-->Add-->Softmax-->Reshape-->MatMul-->Transpose
def get_tmrdasrmt(model):
    res = -1

    node_list= []

    for node in model.graph.node:
        skip = False
        if node.op_type == 'Transpose':
            input_info = next((x for x in model.graph.value_info if x.name == node.input[0]), None)

            if input_info:
                input_dtype = input_info.type.tensor_type.elem_type

                if input_dtype == onnx.TensorProto.FLOAT16:
                    #print(f"Input '{node.input[0]}' of operator '{node.name}' is of type FP16.")
                    logger.debug("Input {} of operator {} is of type FP16.".format(node.input[0], node.name))
                else:
                    #print(f"Input '{node.input[0]}' of operator '{node.name}' is not of type FP16.")
                    continue
            else:
                #print(f"Could not find information for input '{node.input[0]}' of operator '{node.name}'.")
                continue

            if len(node_list) > 0:
                for d in node_list:
                    for _, v in d.items():
                        #print('---k:{}, v:{}'.format(k, v.name))
                        if v.name == node.name:
                            logger.debug('node: {} has been processed'.format(node.name))
                            print('node: {} has been processed'.format(node.name))
                            skip = True
                            break

            if skip == True:
                logger.debug('node: {} has been processed, skip it'.format(node.name))
                continue

            mm_node1, ok = get_next_node_by_output(model, node.output[0])
            if ok == 0 and mm_node1.op_type == 'MatMul':
                logger.debug('got matmul_node: {}'.format(mm_node1.name))
                reshape_node1, ok = get_next_node_by_output(model, mm_node1.output[0])
                if ok == 0 and reshape_node1.op_type == 'Reshape':
                    div_node, ok = get_next_node_by_output(model, reshape_node1.output[0])
                    if ok == 0 and div_node.op_type == 'Div':
                        add_node, ok = get_next_node_by_output(model, div_node.output[0])
                        if ok == 0 and add_node.op_type == 'Add':
                            softmax_node, ok = get_next_node_by_output(model, add_node.output[0])
                            if ok == 0 and softmax_node.op_type == 'Softmax':
                                reshape_node2, ok = get_next_node_by_output(model, softmax_node.output[0])
                                if ok == 0 and reshape_node2.op_type == 'Reshape':
                                    mm_node2, ok = get_next_node_by_output(model, reshape_node2.output[0])
                                    if ok ==0 and mm_node2.op_type == 'MatMul':
                                        #####
                                        mm1_inputB = node
                                        mm1_another_input = mm_node1.input[0]
                                        if mm_node1.input[0] == node.output[0]:
                                            mm1_another_input = mm_node1.input[1]

                                        mm1_another_input_node, _ = get_prev_node_by_input(model, mm1_another_input)
                                        if mm1_another_input_node.op_type == 'Transpose':
                                            mm1_inputA = mm1_another_input_node

                                            if mm_node1.input[0] == node.output[0]:
                                                mm1_inputA = node
                                                mm1_inputB = mm1_another_input_node

                                            mm2_another_input = mm_node2.input[1]
                                            if mm_node2.input[1] == reshape_node2.output[0]:
                                                mm2_another_input = mm_node2.input[0]

                                            mm2_another_input_node, _ = get_prev_node_by_input(model, mm2_another_input)
                                            if mm2_another_input_node.op_type == 'Transpose':
                                                logger.debug('xxxx got match Transpose node: {}'.format(node.name))

                                                res = 0

                                                node_dict = {}
                                                #node_dict['tp'] = node
                                                node_dict['mm1'] = mm_node1
                                                node_dict['mm1_inputA'] = mm1_inputA
                                                node_dict['mm1_inputB'] = mm1_inputB
                                                node_dict['softmax'] = softmax_node
                                                node_dict['rs1'] = reshape_node1
                                                node_dict['rs2'] = reshape_node2
                                                node_dict['add'] = add_node
                                                node_dict['div'] = div_node
                                                node_dict['mm2'] = mm_node2
                                                node_dict['mm2_inputB'] = mm2_another_input_node

                                                node_list.append(node_dict)

                                                '''
                                                print('tp:', node.name)
                                                print('mm1:', mm_node1.name)
                                                print('mm1_inputA:', mm1_inputA.name)
                                                print('mm1_inputB:', mm1_inputB.name)
                                                print('softmax_node:', softmax_node.name)
                                                print('mm2:', mm_node2.name)
                                                '''


    return node_list, res

def handle_add_reshape_block(model, transpose_node):
    d = {}

    rs_node, _ = operation.get_prev_node_by_input(model, transpose_node.input[0])
    add_node, _ =  operation.get_prev_node_by_input(model, rs_node.input[0])
    concat_node, _ = operation.get_prev_node_by_input(model, rs_node.input[1])

    us_node0, _ = operation.get_prev_node_by_input(model, concat_node.input[0])
    us_node1, _ = operation.get_prev_node_by_input(model, concat_node.input[1])

    gather_node0,_ = operation.get_prev_node_by_input(model, us_node0.input[0])

    mul_node, _ = operation.get_prev_node_by_input(model, us_node1.input[0])
    gather_node1,_ = operation.get_prev_node_by_input(model, mul_node.input[0])

    shape_node,_ = operation.get_prev_node_by_input(model, gather_node1.input[0])

    mul_b_name = mul_node.input[1]
    mul_b_input = next(initializer for initializer in model.graph.initializer if initializer.name == mul_b_name)
    mul_b = numpy_helper.to_array(mul_b_input).item()

    add_a_name = add_node.input[0]
    add_a_input = next(initializer for initializer in model.graph.initializer if initializer.name == add_a_name)
    add_a = numpy_helper.to_array(add_a_input)

    #shape = values.get_tensor_shape_by_name(model, shape_node.input[0])

    print('mul_b:', mul_b)
    print('add_a:', add_a.shape[0], add_a.shape[0]//mul_b)

    add_node.output[0] = rs_node.output[0]

    operation.remove_onnx_node(model, rs_node)
    operation.remove_onnx_node(model, concat_node)
    operation.remove_onnx_node(model, us_node0)
    operation.remove_onnx_node(model, us_node1)
    operation.remove_onnx_node(model, gather_node0)
    operation.remove_onnx_node(model, gather_node1)
    operation.remove_onnx_node(model, mul_node)
    operation.remove_onnx_node(model, shape_node)

    '''
    d['rs'] = rs_node
    d['add'] = add_node
    d['concat'] = concat_node
    d['us0'] = us_node0
    d['us1'] = us_node1
    d['gather0'] = gather_node0
    d['gather1'] = gather_node1
    d['mul'] = mul_node
    d['shape'] = shape_node
    d['mul_b'] = mul_b

    print('rs:{}'.format(d['rs'].name))
    print('add:{}'.format(d['add'].name))
    print('concat:{}'.format(d['concat'].name))
    print('us0:{}'.format(d['us0'].name))
    print('us1:{}'.format(d['us1'].name))
    print('gather0:{}'.format(d['gather0'].name))
    print('gather1:{}'.format(d['gather1'].name))
    print('mul:{}'.format(d['mul'].name))
    print('shape:{}'.format(d['shape'].name))
    print('add2:{}'.format(d['add2'].name))
    '''

    return int(add_a.shape[0]//mul_b), int(mul_b)

def handle_mha_block_pattern_six(model):
    node_list, ok = get_tmrdasrmt(model)
    if ok == 0:
        index = 0
        for node_dict in node_list:
            matmul_node1 = node_dict['mm1']
            matmul_node2 = node_dict['mm2']
            mm1_inputA = node_dict['mm1_inputA']
            mm1_inputB = node_dict['mm1_inputB']
            softmax_node = node_dict['softmax']
            mm2_inputB = node_dict['mm2_inputB']

            reshape_node1 = node_dict['rs1']
            reshape_node2 = node_dict['rs2']
            div_node = node_dict['div']
            add_node = node_dict['add']

            logger.debug('start handle tmrdasrmt, matmul name:{}'.format(matmul_node1.name))

            mha_node_name = 'MultiHeadAttention_pattern6_' + str(index)
            index = index + 1

            mha_output_name = mha_node_name + '_output_'

            q_output_shape = values.get_tensor_shape_by_name(model, mm1_inputA.input[0])
            k_output_shape = q_output_shape #values.get_tensor_shape_by_name(model, tp_node2.input[0])
            v_output_shape = q_output_shape #values.get_tensor_shape_by_name(model, tp_node2.input[0])

            logger.debug('==== q_shape:{}, k_shape:{}, v_shape:{}'.format(q_output_shape, k_output_shape, v_output_shape))

            #mha_output_shape = q_output_shape
            #mha_output = onnx.helper.make_tensor_value_info(mha_output_name, onnx.TensorProto.FLOAT16, mha_output_shape)

            #####
            head_dim_ = -1
            num_heads_ = -1
            num_kv_heads_ = -1

            if len(q_output_shape) >= 2:
                head_dim_ = q_output_shape[-1]
                num_heads_ = q_output_shape[-2]

            if len(k_output_shape) >= 2:
                num_kv_heads_ = k_output_shape[-2]
            ####
            if head_dim_ > 256:
                return

            head_dim_, num_heads_ =handle_add_reshape_block(model, mm1_inputA)
            head_dim_, num_kv_heads_ = handle_add_reshape_block(model, mm1_inputB)
            handle_add_reshape_block(model, mm2_inputB)
            # import pdb;pdb.set_trace()
            print('head_dim_:{}, num_heads_:{}'.format(head_dim_, num_heads_))


            input3 = add_node.input[1]
            if add_node.input[1] == div_node.output[0]:
                input3 = add_node.input[0]

            mha_node = onnx.helper.make_node(
                                                'MultiHeadAttentionV1',
                                                name=mha_node_name,
                                                inputs=[mm1_inputA.input[0], mm1_inputB.input[0], mm2_inputB.input[0], input3],
                                                outputs=[mha_output_name],
                                                head_dim=head_dim_, #q_output_shape[-1],
                                                is_causal=0,
                                                num_heads=num_heads_, #q_output_shape[-2],
                                                num_kv_heads=num_kv_heads_, #k_output_shape[-2],
                                                domain='com.metax-tech')

            #model.graph.value_info.append(mha_output)

            insert_node(model, mha_node, mm1_inputA)

            tp_node2, _ = operation.get_next_node_by_output(model, matmul_node2.output[0])

            next_node, _ = get_next_node_by_output(model, tp_node2.output[0])
            for i, input_ in enumerate(next_node.input):
                if tp_node2.output[0] == input_:
                    next_node.input[i] = mha_output_name

            #operation.remove_onnx_node(model, tp_node)
            operation.remove_onnx_node(model, mm1_inputA)
            operation.remove_onnx_node(model, mm1_inputB)
            operation.remove_onnx_node(model, matmul_node1)
            operation.remove_onnx_node(model, reshape_node1)
            operation.remove_onnx_node(model, div_node)
            operation.remove_onnx_node(model, add_node)
            operation.remove_onnx_node(model, softmax_node)
            operation.remove_onnx_node(model, reshape_node2)
            operation.remove_onnx_node(model, matmul_node2)
            operation.remove_onnx_node(model, mm2_inputB)
            operation.remove_onnx_node(model, tp_node2)

        handle_unused_node_list(model)
        del_unused_concat(model)

#################
################ pattern SEVEN
def match_mha_block_pattern_seven(model):
    res = -1

    for node in model.graph.node:
        if node.op_type == 'Transpose':
            #print('node.name:', node.name)
            node_list = get_node_group(model, node.input[0], 6, [0,0,0,0,0,0])
            if len(node_list) == 6:
                expected_pattern = ['Reshape', 'MatMul', 'Softmax', 'Add', 'Mul', 'MatMul']
                for idx1, n in enumerate(node_list):
                    #print('-- node:', idx1, n.op_type, expected_pattern[idx1])
                    if n.op_type != expected_pattern[idx1]:
                        #print('xxxx  node:', idx1, n.op_type, expected_pattern[idx1])
                        break

                if idx1 == 5:
                    res = 0
                    print('match_mha_block_pattern_seven, got it!!!!')
                    break
    return res

###
#Transpose-->Reshape-->Transpose-->MatMul-->Mul-->Add-->Softmax-->MatMul-->Reshape
def get_trtmmasmr(model):
    res = -1

    node_list= []

    for node in model.graph.node:
        skip = False
        if node.op_type == 'Transpose':
            input_info = next((x for x in model.graph.value_info if x.name == node.input[0]), None)

            if input_info:
                input_dtype = input_info.type.tensor_type.elem_type

                if input_dtype == onnx.TensorProto.FLOAT16:
                    #print(f"Input '{node.input[0]}' of operator '{node.name}' is of type FP16.")
                    logger.debug("Input {} of operator {} is of type FP16.".format(node.input[0], node.name))
                else:
                    #print(f"Input '{node.input[0]}' of operator '{node.name}' is not of type FP16.")
                    continue
            else:
                #print(f"Could not find information for input '{node.input[0]}' of operator '{node.name}'.")
                continue

            if len(node_list) > 0:
                for d in node_list:
                    for _, v in d.items():
                        #print('---k:{}, v:{}'.format(k, v.name))
                        if v.name == node.name:
                            logger.debug('node: {} has been processed'.format(node.name))
                            #print('node: {} has been processed'.format(node.name))
                            skip = True
                            break

            if skip == True:
                logger.debug('node: {} has been processed, skip it'.format(node.name))
                continue

            rs_node1, ok = get_next_node_by_output(model, node.output[0])
            if ok == 0 and rs_node1.op_type == 'Reshape':
                all_next_node, ok = get_all_next_node_by_output(model, rs_node1.output[0])
                if ok == 0 and len(all_next_node) == 2:
                    for tp_node1 in all_next_node:
                        if tp_node1.op_type == 'Transpose':
                            mm_node1, ok = get_next_node_by_output(model, tp_node1.output[0])
                            if ok == 0 and mm_node1.op_type == 'MatMul':
                                logger.debug('got matmul_node: {}'.format(mm_node1.name))
                                mul_node, ok = get_next_node_by_output(model, mm_node1.output[0])
                                if ok == 0 and mul_node.op_type == 'Mul':
                                    add_node, ok = get_next_node_by_output(model, mul_node.output[0])
                                    if ok == 0 and add_node.op_type == 'Add':
                                        softmax_node, ok = get_next_node_by_output(model, add_node.output[0])
                                        if ok == 0 and softmax_node.op_type == 'Softmax':
                                            mm_node2, ok = get_next_node_by_output(model, softmax_node.output[0])
                                            if ok ==0 and mm_node2.op_type == 'MatMul':
                                                #####
                                                mm1_inputB = tp_node1
                                                mm1_another_input = mm_node1.input[0]
                                                if mm_node1.input[0] == node.output[0]:
                                                    mm1_another_input = mm_node1.input[1]

                                                mm1_another_input_node, _ = get_prev_node_by_input(model, mm1_another_input)
                                                if mm1_another_input_node.op_type == 'Reshape':
                                                    mm1_inputA = mm1_another_input_node

                                                    if mm_node1.input[0] == node.output[0]:
                                                        mm1_inputA = node
                                                        mm1_inputB = mm1_another_input_node

                                                    mm2_another_input = mm_node2.input[1]
                                                    if mm_node2.input[1] == softmax_node.output[0]:
                                                        mm2_another_input = mm_node2.input[0]

                                                    mm2_another_input_node, _ = get_prev_node_by_input(model, mm2_another_input)
                                                    if mm2_another_input_node.op_type == 'Reshape':
                                                        logger.debug('xxxx got match Transpose node: {}'.format(node.name))

                                                        res = 0

                                                        node_dict = {}
                                                        node_dict['tp1'] = node
                                                        node_dict['mm1'] = mm_node1
                                                        node_dict['mm1_inputA'] = mm1_inputA
                                                        node_dict['mm1_inputB'] = mm1_inputB
                                                        node_dict['tp2'], _ = get_prev_node_by_input(model, mm1_inputA.input[0])
                                                        node_dict['softmax'] = softmax_node
                                                        node_dict['rs1'] = rs_node1
                                                        node_dict['add'] = add_node
                                                        node_dict['mul'] = mul_node
                                                        node_dict['mm2'] = mm_node2
                                                        node_dict['mm2_inputB'] = mm2_another_input_node
                                                        node_dict['tp3'], _ = get_prev_node_by_input(model, mm2_another_input_node.input[0])

                                                        node_list.append(node_dict)

                                                        '''
                                                        print('tp1:', node.name)
                                                        print('mm1:', mm_node1.name)
                                                        print('mm1_inputA:', mm1_inputA.name)
                                                        print('mm1_inputB:', mm1_inputB.name)
                                                        print('softmax_node:', softmax_node.name)
                                                        print('mm2:', mm_node2.name)
                                                        '''

                            break

    return node_list, res

#Shape-->Gather-->Unsqueeze-->Concat-->ConstantOfShape-->Mul
def get_sguccm(model):
    node_list = []
    for node in model.graph.node:
        if node.op_type == 'Shape':
            gather_node, ok = get_next_node_by_output(model, node.output[0])
            if ok == 0 and gather_node.op_type == 'Gather':
                us_node, ok = get_next_node_by_output(model, gather_node.output[0])
                if ok == 0 and us_node.op_type == 'Unsqueeze':
                    concat_node, ok = get_next_node_by_output(model, us_node.output[0])
                    if ok == 0 and concat_node.op_type == 'Concat':
                        cos_node, ok = get_next_node_by_output(model, concat_node.output[0])
                        if ok == 0 and cos_node.op_type == 'ConstantOfShape':
                            mul_node, ok = get_next_node_by_output(model, cos_node.output[0])
                            if ok == 0 and mul_node.op_type == 'Mul':
                                us_node1, _ = get_prev_node_by_input(model, concat_node.input[0])
                                gather_node1, _ = get_prev_node_by_input(model, us_node1.input[0])
                                shape_node1, _ = get_prev_node_by_input(model, gather_node1.input[0])

                                us_node2, _ = get_prev_node_by_input(model, concat_node.input[1])
                                gather_node2, _ = get_prev_node_by_input(model, us_node2.input[0])

                                us_node3, _ = get_prev_node_by_input(model, concat_node.input[2])
                                gather_node3, _ = get_prev_node_by_input(model, us_node3.input[0])
                                shape_node2, _ = get_prev_node_by_input(model, gather_node3.input[0])

                                found = False

                                for d in node_list:
                                    if mul_node.name == d['Mul'].name:
                                        #print('mul node {} has been processed'.format(mul_node.name))
                                        found = True
                                        break

                                if found == False:
                                    d = {}
                                    d['Mul'] = mul_node
                                    d['COS'] = cos_node
                                    d['Concat'] = concat_node

                                    print('got Concat node:', concat_node.name)

                                    d['us1'] = us_node1
                                    d['gather1'] = gather_node1
                                    d['shape1'] = shape_node1

                                    d['us2'] = us_node2
                                    d['gather2'] = gather_node2

                                    d['us3'] = us_node3
                                    d['gather3'] = gather_node3
                                    d['shape2'] = shape_node2

                                    node_list.append(d)

    return node_list

def handle_sguccm(model):
    node_list = get_sguccm(model)
    for d in node_list:
        mul_node = d['Mul']
        cos_node = d['COS']
        concat_node = d['Concat']

        us_node1 = d['us1']
        gather_node1 = d['gather1']
        shape_node1 = d['shape1']

        us_node2 = d['us2']
        gather_node2 = d['gather2']

        us_node3 = d['us3']
        gather_node3 = d['gather3']
        shape_node2 = d['shape2']

        #print('mul_node:', mul_node.name)

        operation.remove_onnx_node(model, mul_node)
        operation.remove_onnx_node(model, cos_node)
        operation.remove_onnx_node(model, concat_node)

        operation.remove_onnx_node(model, us_node1)
        operation.remove_onnx_node(model, gather_node1)
        operation.remove_onnx_node(model, shape_node1)

        operation.remove_onnx_node(model, us_node2)
        operation.remove_onnx_node(model, gather_node2)

        operation.remove_onnx_node(model, us_node3)
        operation.remove_onnx_node(model, gather_node3)
        operation.remove_onnx_node(model, shape_node2)

def handle_shape_input(model, mha_node):
    input2 = mha_node.input[2]
    reshape_node, _ = get_prev_node_by_input(model, input2)

    reshape_node2, _ = get_next_node_by_output(model, mha_node.output[0])
    concat_node, _ = get_prev_node_by_input(model, reshape_node2.input[1])
    us_node, _ = get_prev_node_by_input(model, concat_node.input[0])
    gather_node, ok = get_prev_node_by_input(model, us_node.input[0])
    if ok == 0 and gather_node.op_type == 'Gather':
        shape_node, ok = get_prev_node_by_input(model, gather_node.input[0])
        if ok == 0 and shape_node.op_type == 'Shape':
            shape_node.input[0] = reshape_node.input[0]
    elif ok == 0 and gather_node.op_type == 'Div':
        gather_node2, ok = get_prev_node_by_input(model, gather_node.input[0])
        if ok == 0 and gather_node2.op_type == 'Gather':
            shape_node, ok = get_prev_node_by_input(model, gather_node2.input[0])
            if ok == 0 and shape_node.op_type == 'Shape':
                #print('old shape_node.input[0]:', shape_node.input[0])

                shape_node.input[0] = reshape_node.input[0]

                #print('new shape_node.input[0]:', shape_node.input[0])

def handle_mha_block_pattern_seven(model):
    node_list, ok = get_trtmmasmr(model)
    if ok == 0:
        index = 0
        for node_dict in node_list:
            matmul_node1 = node_dict['mm1']
            matmul_node2 = node_dict['mm2']
            mm1_inputA = node_dict['mm1_inputA']
            mm1_inputB = node_dict['mm1_inputB']
            softmax_node = node_dict['softmax']
            mm2_inputB = node_dict['mm2_inputB']

            reshape_node1 = node_dict['rs1']
            mul_node = node_dict['mul']
            add_node = node_dict['add']
            tp_node1 = node_dict['tp1']
            tp_node2 = node_dict['tp2']
            tp_node3 = node_dict['tp3']

            logger.debug('start handle trtmmasmr, matmul name:{}'.format(matmul_node1.name))

            mha_node_name = 'MultiHeadAttention_pattern7_' + str(index)
            index = index + 1

            mha_output_name = mha_node_name + '_output_'

            q_output_shape = values.get_tensor_shape_by_name(model, mm1_inputA.input[0])
            k_output_shape = q_output_shape #values.get_tensor_shape_by_name(model, tp_node2.input[0])
            v_output_shape = q_output_shape #values.get_tensor_shape_by_name(model, tp_node2.input[0])

            logger.debug('==== q_shape:{}, k_shape:{}, v_shape:{}'.format(q_output_shape, k_output_shape, v_output_shape))

            #mha_output_shape = q_output_shape
            #mha_output = onnx.helper.make_tensor_value_info(mha_output_name, onnx.TensorProto.FLOAT16, mha_output_shape)

            #####
            head_dim_ = -1
            num_heads_ = -1
            num_kv_heads_ = -1

            if len(q_output_shape) >= 2 and q_output_shape[-1] > 0:
                head_dim_ = q_output_shape[-1]
                num_heads_ = q_output_shape[-2]

            if len(k_output_shape) >= 2 and k_output_shape[-2] > 0:
                num_kv_heads_ = k_output_shape[-2]
            ####
            if head_dim_ > 256:
                return

            input3 = add_node.input[1]
            if add_node.input[1] == mul_node.output[0]:
                input3 = add_node.input[0]

            mha_node = onnx.helper.make_node(
                                                'MultiHeadAttentionV1',
                                                name=mha_node_name,
                                                inputs=[tp_node2.input[0], tp_node1.input[0], tp_node3.input[0]],
                                                outputs=[mha_output_name],
                                                head_dim=head_dim_, #q_output_shape[-1],
                                                is_causal=0,
                                                num_heads=num_heads_, #q_output_shape[-2],
                                                num_kv_heads=num_kv_heads_, #k_output_shape[-2],
                                                domain='com.metax-tech')

            #model.graph.value_info.append(mha_output)

            insert_node(model, mha_node, mm1_inputA)

            reshape_node2 = None
            tp_node4 = None

            all_next_node, _ = operation.get_all_next_node_by_output(model, matmul_node2.output[0])
            for n in all_next_node:
                if n.op_type == 'Reshape':
                    reshape_node2 = n
                    tp_node4, _ = get_next_node_by_output(model, n.output[0])
                    next_node, _ = get_next_node_by_output(model, tp_node4.output[0])
                    for i, input_ in enumerate(next_node.input):
                        if tp_node4.output[0] == input_:
                            next_node.input[i] = mha_output_name

            operation.remove_onnx_node(model, tp_node1)
            operation.remove_onnx_node(model, mm1_inputA)
            operation.remove_onnx_node(model, mm1_inputB)
            operation.remove_onnx_node(model, matmul_node1)
            operation.remove_onnx_node(model, reshape_node1)
            operation.remove_onnx_node(model, mul_node)
            operation.remove_onnx_node(model, add_node)
            operation.remove_onnx_node(model, softmax_node)
            operation.remove_onnx_node(model, reshape_node2)
            operation.remove_onnx_node(model, matmul_node2)
            operation.remove_onnx_node(model, mm2_inputB)
            operation.remove_onnx_node(model, tp_node2)
            operation.remove_onnx_node(model, tp_node3)
            operation.remove_onnx_node(model, tp_node4)

            handle_shape_input(model, mha_node)

        handle_unused_node_list(model)
        del_unused_concat(model)
        del_unused_unsqueeze(model)
        #handle_shape_input(model, mha_node)
        handle_sguccm(model)

def mha_fuse(model):
    pattern = -1
    ret1 = match_mha_block_pattern_one(model) #for decoder_model_bs10_sim_fp16.onnx
    ret2 = match_mha_block_pattern_two(model) #for bert_cls.onnx
    if ret2 == -1:
        ret2 = match_mha_block_pattern_two_ext(model) #for bert_squad_v1.onnx

    ret3 = match_mha_block_pattern_three(model) #for unet_mha.onnx
    ret4 = match_mha_block_pattern_four(model) #for text_encoder.onnx
    ret5 = match_mha_block_pattern_five(model) #for model_fix.onnx unet
    ret6 = -1 #match_mha_block_pattern_six(model) #for bert-base-f32-squad_fp16.onnx
    ret7 = match_mha_block_pattern_seven(model) #for vae decoder

    #print('zzzzzzzzzzzzzzzz ret1: {}, ret2: {}, ret3:{}, ret4: {}'.format(ret1, ret2, ret3, ret4))

    if ret1 == 0:
        pattern = 1
        handle_mha_block_pattern_one(model)

    if ret2 == 0:
        pattern = 2
        handle_mha_block_pattern_two(model)

    if ret3 == 0:
        pattern = 3
        handle_mha_block_pattern_three(model)

    if ret4 == 0:
        pattern = 4
        handle_mha_block_pattern_four(model)

    if ret5 == 0:
        pattern = 5
        handle_mha_block_pattern_five(model)

    if ret6 == 0:
        pattern = 6
        handle_mha_block_pattern_six(model)

    if ret7 == 0:
        pattern = 7
        handle_mha_block_pattern_seven(model)

    print('got pattern = ', pattern)

    if pattern == -1:
        logger.debug('This is not a mha model---')
        return model

    op_set = model.opset_import.add()
    op_set.domain = 'com.metax-tech'
    op_set.version = 1

    return model

'''
if __name__ == "__main__":
    model = onnx.load('/home/zqiu/models/decoder_model_bs10_fp16.onnx')
    mha_fuse(model)
    onnx.save(model, './mha.onnx')
'''
