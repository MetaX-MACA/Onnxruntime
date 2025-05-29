import onnx
#import re
import operation

'''
def extract_number(text):
    match = re.search(r'_(\d+)$', text)
    if match:
        return int(match.group(1))
    else:
        return -1
'''

def naming_onnx_node(model):
    op_type_name_map = {}
    null_name_node_map = {}

    for node in model.graph.node:
        if node.name != '':
            if node.op_type in op_type_name_map.keys():
                op_type_name_map[node.op_type].append(node.name)
            else:
                op_type_name_map[node.op_type] = [node.name]
        else:
            if node.op_type in null_name_node_map.keys():
                null_name_node_map[node.op_type].append(node)
            else:
                null_name_node_map[node.op_type] = [node]

    if len(op_type_name_map) == 0: 
        max_num = 0
        for node in model.graph.node:
            node.name = node.op_type + '_' + str(max_num)
            #print('new node.name:', node.name )
            max_num = max_num + 1 

        '''
        for op_type, node_list in null_name_node_map.items():
            max_num = -1

            for node in node_list:
                node.name = op_type + '_' + str(max_num + 1)
                #print('new node.name:', node.name )
                max_num = max_num + 1
        '''
    else:
        max_num = 0
        for _, node_list in null_name_node_map.items():
            for node in node_list:
                prev_name = ''
                next_name = ''
                prev_node = None
                next_node = None

                if len(node.input):
                    prev_node, _ = operation.get_prev_node_by_input(model, node.input[0])
                    prev_name = prev_node.name

                if len(node.output):    
                    next_node, _ = operation.get_next_node_by_output(model, node.output[0])
                    next_name = next_node.name

                if prev_name == '':
                    if prev_node != None:
                        prev_name = prev_node.op_type + '_' + str(max_num)
                    else:
                        prev_name = node.op_type + '_prev_' + str(max_num)   
                    max_num = max_num + 1

                if next_name == '':
                    if next_node != None:
                        next_name = next_node.op_type + '_' + str(max_num)
                    else:
                        next_name = node.op_type + '_post_' + str(max_num)    
                    max_num = max_num + 1

                print('prev_node.name:', prev_name)
                print('next_node.name:', next_name)

                node.name = prev_name + '#' + node.op_type + str(max_num) + '#' + next_name
                max_num = max_num + 1

                #node.name = prev_node.name + '#' #+ node.op_type + '#' + next_node.name   

    return model            

'''         
model = onnx.load('/home/zqiu/models/yolov2-coco-9_dy.onnx')

naming_onnx_node(model)

onnx.save(model, './test.onnx')
'''
