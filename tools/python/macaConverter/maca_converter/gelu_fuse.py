import onnx
import operation
import values
import numpy as np
import log

logger = log.getLogger(__name__, log.INFO)

def merge_gelu1(model):
    dict_div = {}
    dict_erf = {}
    dict_add = {}
    dict_mul = {}
    dict_mul2 = {}

    got_gelu = False

    search = True

    while search == True:
        search = False
        got_match_mul = False

        for node_id, node in enumerate(model.graph.node):
            #print(node_id, ", name:", node.name, ", input:", node.input, ", output:", node.output,  \
            #        ", op:", node.op_type, ', len(input):', len(node.input))

            if node.op_type == 'Div':
                divB = values.get_init_value(model, node.input[1])
                logger.debug('divB: {}'.format(divB))

                if isinstance(divB, list) and divB == []:
                    logger.debug('divB is not in initilizer')
                    #continue
                    divB = values.get_constant_value(model, node.input[1])
                    if divB == []:
                        logger.debug('divB is not in constant node list')
                        got_match_mul = False
                        continue
                    else:
                        logger.debug('divB is {} {}'.format(divB, type(divB)))    

                if abs(divB[0] - 1.414) > 0.01:
                    logger.debug('this is not the div-node which we wanted(value B is not 1.414)...')
                    got_match_mul = False
                    continue

                if isinstance(divB, np.ndarray) == True:
                    if divB.shape != (1, ):
                        logger.debug('this is not the div-node which we wanted(shape is wrong)...')
                        got_match_mul = False
                        continue
                else:        
                    if len(divB) != 1:
                        logger.debug('this is not the div-node which we wanted(list len is wrong)...')
                        got_match_mul = False
                        continue

                dict_div['input'] = node.input
                dict_div['output'] = node.output
                dict_div['id'] = node_id
                logger.debug('got match div node: {}'.format(node.name))

            if node.op_type == 'Erf':
                if dict_div and node.input[0] == dict_div['output'][0]:
                    dict_erf['input'] = node.input
                    dict_erf['output'] = node.output
                    dict_erf['id'] = node_id

                    logger.debug('got first pair: {} {}'.format(dict_erf['input'], dict_erf['output']))
                else:
                    logger.debug('clear dict_div: {}'.format(dict_div))
                    got_match_mul = False
                    dict_div = {}    

            if node.op_type == 'Add':
                if dict_erf and node.input[0] == dict_erf['output'][0]:
                    addB = values.get_init_value(model, node.input[1])
                    if isinstance(addB, list) and addB == []:
                        logger.debug('addB is not in initilizer')
                        addB = values.get_constant_value(model, node.input[1])
                        if addB == []:
                            dict_div = {}
                            dict_erf = {}
                            got_match_mul = False
                            logger.debug('addB is not in constant node list~')
                            continue

                    logger.debug('addB: {}'.format(addB))

                    if abs(addB[0] - 1) > 0.01:
                        logger.debug('this is not the add-node which we wanted(value B is not 1)...')
                        got_match_mul = False
                        dict_div = {}
                        dict_erf = {}
                        continue

                    if isinstance(addB, np.ndarray) == True:
                        if addB.shape != (1, ):
                            logger.debug('this is not the add-node which we wanted(shape is wrong)...')
                            dict_div = {}
                            dict_erf = {}
                            got_match_mul = False
                            continue
                    else:        
                        if len(addB) != 1:
                            logger.debug('this is not the add-node which we wanted(list len is wrong)...')
                            dict_div = {}
                            dict_erf = {}
                            got_match_mul = False
                            continue

                    dict_add['input'] = node.input
                    dict_add['output'] = node.output
                    dict_add['id'] = node_id                
                else:
                    logger.debug('clear dict_add and dict_erf, ')
                    logger.debug('dict_add: {}'.format(dict_add))
                    logger.debug('dict_erf: {}'.format(dict_erf))
                    got_match_mul = False
                    dict_div = {}
                    dict_erf = {}

            if node.op_type == 'Mul':
                if got_match_mul == False and dict_div and node.input[0] == dict_div['input'][0] and \
                        node.input[1] == dict_add['output'][0]:
                    dict_mul['input'] = node.input
                    dict_mul['output'] = node.output
                    dict_mul['id'] = node_id

                    got_match_mul = True

                    logger.debug('got second pair: {} {}'.format(dict_mul['input'], dict_mul['output']))
                elif got_match_mul == True:
                    if node.input[0] == dict_mul['output'][0]:
                        mulB = values.get_init_value(model, node.input[1])
                        if isinstance(mulB, list) and mulB == []:
                            logger.debug('mulB is not in initilizer')
                            mulB = values.get_constant_value(model, node.input[1])
                            if mulB == []:
                                dict_div = {}
                                dict_erf = {}
                                dict_add = {}
                                dict_mul = {}
                                got_match_mul = False
                                logger.debug('mulB is not in constant node list~')
                                continue

                        logger.debug('mulB: {}'.format(mulB))

                        if abs(mulB[0] - 0.5) > 0.01:
                            logger.debug('this is not the mul-node which we wanted(value B is not 1)...')
                            got_match_mul = False
                            dict_div = {}
                            dict_erf = {}
                            dict_add = {}
                            dict_mul = {}
                            continue

                        if isinstance(mulB, np.ndarray) == True:
                            if mulB.shape != (1, ):
                                logger.debug('this is not the mul-node which we wanted(shape is wrong)...')
                                dict_div = {}
                                dict_erf = {}
                                dict_add = {}
                                dict_mul = {}
                                got_match_mul = False
                                continue
                        else:        
                            if len(mulB) != 1:
                                logger.debug('this is not the mul-node which we wanted(list len is wrong)...')
                                dict_div = {}
                                dict_erf = {}
                                dict_add = {}
                                dict_mul = {}
                                got_match_mul = False
                                continue

                        dict_mul2['input'] = node.input
                        dict_mul2['output'] = node.output
                        dict_mul2['id'] = node_id

                        ###################################
                        old_node = model.graph.node[dict_div['id']] 
                        model.graph.node.remove(old_node)

                        gelu_node = onnx.helper.make_node(
                                                name = '',
                                                op_type='Gelu',
                                                inputs=[dict_div['input'][0]],
                                                outputs=dict_mul2['output'],
                                                domain='com.microsoft'
                                                )

                        model.graph.node.insert(dict_div['id'], gelu_node)

                        operation.remove_node(model, dict_erf['input'], dict_erf['output'])
                        operation.remove_node(model, dict_add['input'], dict_add['output'])
                        operation.remove_node(model, dict_mul['input'], dict_mul['output'])
                        operation.remove_node(model, dict_mul2['input'], dict_mul2['output'])

                        dict_div = {}
                        dict_erf = {}
                        dict_add = {}
                        dict_mul = {}
                        dict_mul2 = {}
                        got_match_mul = False
                        ###############################
                        got_gelu = True
                        search = True
                        break           
                    else:
                        logger.debug('----clear dict_div, dict_erf, dict_mul')
                        dict_div = {}
                        dict_erf = {}
                        dict_add = {}
                        got_match_mul = False             
                else:
                    logger.debug('+++clear dict_div, dict_erf, dict_mul')
                    dict_div = {}
                    dict_erf = {}
                    dict_add = {}
                    dict_mul = {}
                    got_match_mul = False


    if got_gelu == True:
        op_set = model.opset_import.add()
        op_set.domain = 'com.microsoft'
        op_set.version = 1

    return model

class MergeGelu():
    def __init__(self, model):
        logger.debug('MergeGelu Init--------------------------')
        self.model = model
        self.got_gelu = False
        self.search = True
        self.loop = 0

        self.dict_pow = {}
        self.dict_mul1 = {}
        self.dict_add1 = {}
        self.dict_mul2 = {}
        self.dict_tanh = {}
        self.dict_add2 = {}
        self.dict_mul3 = {}
        self.dict_mul4 = {}

        self.got_mul1 = False
        self.got_mul2 = False
        self.got_mul3 = False
        self.got_mul4 = False

        self.got_add1 = False
        self.got_add2 = False
        
    def clear(self):
        self.dict_pow = {}
        self.dict_mul1 = {}
        self.dict_add1 = {}
        self.dict_mul2 = {}
        self.dict_tanh = {}
        self.dict_add2 = {}
        self.dict_mul3 = {}
        self.dict_mul4 = {}

        self.got_mul1 = False
        self.got_mul2 = False
        self.got_mul3 = False
        self.got_mul4 = False

        self.got_add1 = False
        self.got_add2 = False

        self.search = False

    def merge(self):
        while self.search == True:
            self.clear()

            self.loop = self.loop + 1

            for node_id, node in enumerate(self.model.graph.node):
                self.loop = self.loop + 1
                #print(node_id, ", name:", node.name, ", input:", node.input, ", output:", node.output,  \
                #        ", op:", node.op_type, ', len(input):', len(node.input))

                if node.op_type == 'Pow':
                    powB = values.get_init_value(self.model, node.input[1])
                    if isinstance(powB, list) and powB == []:
                        logger.debug('powB is not in initilizer')
                        powB = values.get_constant_value(self.model, node.input[1])
                        if powB == []:
                            logger.debug('powB is not in constant node list')
                            self.clear()
                            continue
                        else:
                            logger.debug('powB is {} {}'.format(powB, type(powB)))    

                    if abs(powB[0] - 3.0) > 0.01:
                        logger.debug('this is not the pow-node which we wanted(value B is not 3.0) {}'.format(powB[0]))
                        self.clear()
                        continue

                    self.dict_pow['input'] = node.input
                    self.dict_pow['output'] = node.output
                    self.dict_pow['id'] = node_id
                    self.dict_pow['name'] = node.name

                    logger.debug('got pow node: {}'.format(node.name))

                if node.op_type == 'Mul':
                    if self.got_mul1 == False:
                        if self.dict_pow and node.input[1] == self.dict_pow['output'][0]:
                            mulA = values.get_init_value(self.model, node.input[0])
                            logger.debug('mulA: {}'.format(mulA))

                            if isinstance(mulA, list) and mulA == []:
                                logger.debug('mulA is not in initilizer')
                                mulA = values.get_constant_value(self.model, node.input[1])
                                if mulA == []:
                                    logger.debug('mulA is not in constant node list')
                                    self.clear()
                                    continue
                                else:
                                    logger.debug('mulA is {} {}'.format(mulA, type(mulA)))    

                            if abs(mulA[0] - 0.0447) > 0.01:
                                logger.debug('this is not the mul-node which we wanted(value B is not 0.0447)...')
                                self.clear()
                                continue

                            self.dict_mul1['input'] = node.input
                            self.dict_mul1['output'] = node.output
                            self.dict_mul1['id'] = node_id
                            self.dict_mul1['name'] = node.name

                            logger.debug('got mul1 node: {}'.format(node.name))

                            self.got_mul1 = True
                        else:
                            logger.debug('---self.clear 1')
                            self.clear()
                    else:
                        if self.got_mul2 == False:
                            if self.dict_add1 and node.input[1] == self.dict_add1['output'][0]:
                                mulA = values.get_init_value(self.model, node.input[0])
                                logger.debug('mulA: {}'.format(mulA))

                                if isinstance(mulA, list) and mulA == []:
                                    logger.debug('mulA is not in initilizer')
                                    mulA = values.get_constant_value(self.model, node.input[1])
                                    if mulA == []:
                                        logger.debug('mulA is not in constant node list')
                                        self.clear()
                                        continue
                                    else:
                                        logger.debug('mulA is {} {}'.format(mulA, type(mulA)))    

                                if abs(mulA[0] - 0.79788) > 0.01:
                                    logger.debug('this is not the mul-node which we wanted(value B is not 0.79788)...')
                                    self.clear()
                                    continue

                                self.dict_mul2['input'] = node.input
                                self.dict_mul2['output'] = node.output
                                self.dict_mul2['id'] = node_id
                                self.dict_mul2['name'] = node.name
                                self.got_mul2 = True

                                logger.debug('got mul2 node: {}'.format(node.name))
                            else:
                                logger.debug('self.clear 2')
                                self.clear()
                        else:
                            if self.got_mul3 == False:
                                if self.dict_add2 and node.input[1] == self.dict_add2['output'][0]:
                                    mulA = values.get_init_value(self.model, node.input[0])
                                    logger.debug('mulA: {}'.format(mulA))

                                    if isinstance(mulA, list) and mulA == []:
                                        logger.debug('mulA is not in initilizer')
                                        mulA = values.get_constant_value(self.model, node.input[1])
                                        if mulA == []:
                                            logger.debug('mulA is not in constant node list')
                                            self.clear()
                                            continue
                                        else:
                                            logger.debug('mulA is {} {}'.format(mulA, type(mulA)))    

                                    if abs(mulA[0] - 0.5) > 0.01:
                                        logger.debug('this is not the mul-node which we wanted(value B is not 0.5)...')
                                        self.clear()
                                        continue

                                    self.dict_mul3['input'] = node.input
                                    self.dict_mul3['output'] = node.output
                                    self.dict_mul3['id'] = node_id
                                    self.dict_mul3['name'] = node.name
                                    self.got_mul3 = True

                                    logger.debug('got mul3 node: {}'.format(node.name))
                                else: 
                                    logger.debug('self.clear 3') 
                                    self.clear()
                            elif self.got_mul4 == False:
                                if self.dict_pow and self.dict_mul3 and node.input[0] == self.dict_pow['input'][0] and \
                                        node.input[1] == self.dict_mul3['output'][0]:
                                    
                                    logger.debug('got gelu node, begin fusing...')

                                    self.dict_mul4['input'] = node.input
                                    self.dict_mul4['output'] = node.output
                                    self.dict_mul4['id'] = node_id
                                    self.dict_mul4['name'] = node.name
                                    self.got_mul4 = True

                                    self.search = True
                                    self.got_gelu = True

                                    pow_node = self.model.graph.node[self.dict_pow['id']]
                                    self.model.graph.node.remove(pow_node)

                                    gelu_node = onnx.helper.make_node(
                                                name = self.dict_pow['name'] + '+' + self.dict_mul1['name'] + '+' + self.dict_add1['name'] + '+' + self.dict_mul2['name'] + '+' + self.dict_tanh['name'] + '+' + self.dict_add2['name'] + '+' + self.dict_mul3['name'] + '+' + self.dict_mul4['name'],
                                                op_type='Gelu',
                                                inputs=[self.dict_pow['input'][0]],
                                                outputs=self.dict_mul4['output'],
                                                domain='com.microsoft'
                                                )

                                    self.model.graph.node.insert(self.dict_pow['id'], gelu_node)

                                    operation.remove_node(self.model, self.dict_mul1['input'], self.dict_mul1['output'])
                                    operation.remove_node(self.model, self.dict_add1['input'], self.dict_add1['output'])
                                    operation.remove_node(self.model, self.dict_mul2['input'], self.dict_mul2['output'])
                                    operation.remove_node(self.model, self.dict_tanh['input'], self.dict_tanh['output'])
                                    operation.remove_node(self.model, self.dict_add2['input'], self.dict_add2['output'])
                                    operation.remove_node(self.model, self.dict_mul3['input'], self.dict_mul3['output'])
                                    operation.remove_node(self.model, self.dict_mul4['input'], self.dict_mul4['output'])
                                    break

                if node.op_type == 'Add':
                    if self.got_add1 == False:
                        if self.dict_pow and self.dict_mul1 and node.input[0] == self.dict_pow['input'][0] and \
                                node.input[1] == self.dict_mul1['output'][0]: 
                            self.dict_add1['input'] = node.input
                            self.dict_add1['output'] = node.output
                            self.dict_add1['id'] = node_id
                            self.dict_add1['name'] = node.name
                            self.got_add1 = True
                            logger.debug('got add1 node: {}'.format(node.name))
                        else:
                            logger.debug('self.clear 4')
                            self.clear()      
                    else:
                        if self.got_add2 == False:
                            if self.dict_tanh and node.input[1] == self.dict_tanh['output'][0]:
                                addA = values.get_init_value(self.model, node.input[0])

                                if isinstance(addA, list) and addA == []:
                                    logger.debug('addA is not in initilizer')
                                    addA = values.get_constant_value(self.model, node.input[1])
                                    if addA == []:
                                        logger.debug('addA is not in constant node list')
                                        self.clear()
                                        continue
                                    else:
                                        logger.debug('addA is {} {}'.format(addA, type(addA)))    

                                if abs(addA[0] - 1.0) > 0.01:
                                    logger.debug('this is not the mul-node which we wanted(value B is not 1.0)...')
                                    self.clear()
                                    continue

                                self.dict_add2['input'] = node.input
                                self.dict_add2['output'] = node.output
                                self.dict_add2['id'] = node_id
                                self.dict_add2['name'] = node.name
                                self.got_add2 = True

                                logger.debug('got add2 node: {}'.format(node.name))
                            else:
                                logger.debug('self.clear 5')
                                self.clear()          
                        else:
                            logger.debug('got add1 and add2 already----')

                if node.op_type == 'Tanh':
                    if self.got_add1 == True and self.got_mul1 == True and self.got_mul2 == True:
                        if self.dict_mul2 and node.input[0] == self.dict_mul2['output'][0]:
                            self.dict_tanh['input'] = node.input
                            self.dict_tanh['output'] = node.output
                            self.dict_tanh['id'] = node_id
                            self.dict_tanh['name'] = node.name
                            logger.debug('got tanh node: {}'.format(node.name))
                        else:
                            logger.debug('self.clear 6')
                            self.clear()      
                    else:
                        logger.debug('self.clear 7')
                        self.clear()

        if self.got_gelu == True:
            op_set = self.model.opset_import.add()
            op_set.domain = 'com.microsoft'
            op_set.version = 1

        return self.model

def merge_gelu(model):
    model = merge_gelu1(model)

    mg = MergeGelu(model)
    model = mg.merge()

    return model

'''
if __name__ == "__main__":
    model = onnx.load('/home/zqiu/models/gelu3.onnx')
    merge_gelu(model)
    onnx.save(model, './gelu3_fuse.onnx')
'''
    