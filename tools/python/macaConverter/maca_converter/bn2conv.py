import onnx
import values
import numpy as np
import log

logger = log.getLogger(__name__, log.INFO)


def bn2conv(model):
    for node in model.graph.node:
        if node.op_type == 'BatchNormalization':
            input_type = onnx.TensorProto.FLOAT
            input_dims = 4
            for vi in model.graph.value_info:
                if node.input[0] == vi.name:
                    input_type = vi.type.tensor_type.elem_type
                    input_dims = len(vi.type.tensor_type.shape.dim)
                    logger.debug('bn2conv, node.name: {} {} {}'.format(node.name, input_type, input_dims))

                    break

            if input_dims < 3:
                continue

            epsilon = 1e-5
            attributes = node.attribute
            for attr in attributes:
                if attr.name == 'epsilon':
                    epsilon = attr.f
                    logger.debug('epsilon: {}'.format(epsilon))

            logger.debug('node.name: {}, node.input: {}, node.output: {}'.format(node.name, node.input, node.output)) 

            input_dict = {}
            for init in model.graph.initializer:
                if init.name in node.input[1:]:
                    v = values.get_init_value(model, init.name)  
                    if isinstance(v, np.ndarray) == False: 
                        v = np.array(v)

                    input_dict[init.name] = v

            #for k, v in input_dict.items():
            #    print('kv', k) 

            alpha = input_dict[node.input[1]]
            beta = input_dict[node.input[2]]
            mean = input_dict[node.input[3]]
            var = input_dict[node.input[4]]

            input_channel = alpha.shape[0]
            logger.debug('input_channel is {}'.format(input_channel))

            d_type = np.float32
            if input_type == onnx.TensorProto.FLOAT16:
                d_type = np.float16

            w = np.ones((input_channel, 1, 1, 1), dtype=d_type) #np.float32)
            if input_dims == 3:
                w = np.ones((input_channel, 1, 1), dtype=d_type) #np.float32)

            b = np.zeros(
                shape=w.shape[1]*input_channel,
                dtype=d_type #np.float32
            )      

            scale = alpha/np.sqrt(var + epsilon)

            if input_dims == 4:
                w = w * scale.reshape([-1, 1, 1, 1]) 
            else:
                w = w * scale.reshape([-1, 1, 1])  

            logger.debug('w.shape: {}'.format(w.shape))

            b = alpha * (b - mean) / np.sqrt(var + epsilon) + beta

            node.op_type = 'Conv'
            del node.attribute[:]
            if input_dims == 4:
                attr = onnx.helper.make_attribute('kernel_shape', [1, 1])
            else:
                attr = onnx.helper.make_attribute('kernel_shape', [1])  

            node.attribute.append(attr)

            if input_dims == 4:
                attr = onnx.helper.make_attribute('dilations', [1, 1])
            else:
                attr = onnx.helper.make_attribute('dilations', [1]) 

            node.attribute.append(attr)  

            if input_dims == 4:
                attr = onnx.helper.make_attribute('strides', [1, 1])
            else:
                attr = onnx.helper.make_attribute('strides', [1])  

            node.attribute.append(attr) 

            if input_dims == 4:
                attr = onnx.helper.make_attribute('pads', [0, 0, 0, 0])
            else:
                attr = onnx.helper.make_attribute('pads', [0, 0]) 

            node.attribute.append(attr)  

            attr = onnx.helper.make_attribute('group', input_channel)
            node.attribute.append(attr)      

            del node.input[1:]

            w_list = w.flatten().tolist()
            b_list = b.flatten().tolist()

            w_var_name = node.name+'_w_'
            b_var_name = node.name+'_b_'

            w_var = onnx.helper.make_tensor(name=w_var_name,
                        data_type=input_type,
                        dims=w.shape,
                        vals=w_list)

            model.graph.initializer.append(w_var)  

            b_var = onnx.helper.make_tensor(name=b_var_name,
                        data_type=input_type,
                        dims=b.shape,
                        vals=b_list)

            model.graph.initializer.append(b_var)  

            node.input.append(w_var_name) 
            node.input.append(b_var_name) 

    #onnx.save(model)  

    return model

'''
model = onnx.load('./test.onnx')
model = onnx.shape_inference.infer_shapes(model)  
bn2conv(model)
'''

