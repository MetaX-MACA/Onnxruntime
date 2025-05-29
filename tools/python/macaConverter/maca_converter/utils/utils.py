import re
import numpy as np
from typing import Union
from copy import deepcopy


def find_node_by_output(model, output):
    for node in model.graph.node:
        if node.op_type == "Constant":
            continue

        if output in node.output:
            return node

def find_node_by_input(model, input):
    for node in model.graph.node:
        if node.op_type == "Constant":
            continue

        if input in node.input:
            return node

def find_nodes_by_input(model, input):
    nodes = []
    for node in model.graph.node:
        if node.op_type == "Constant":
            continue

        if input in node.input:
            nodes.append(node)
    return nodes

def find_nodes_by_output(model, output):
    nodes = []
    for node in model.graph.node:
        if node.op_type == "Constant":
            continue

        if output in node.output:
            nodes.append(node)
    return nodes

def find_consts(model, name):
    nodes = []
    for node in model.graph.node:
        if name in node.output and node.op_type == "Constant":
            nodes.append(node)
    return nodes

def find_initializers(model, name):
    nodes = []
    for node in model.graph.initializer:
        if node.name == name:
            nodes.append(node)
    return nodes


def find_initializers_input(model, node):
    const_init = []
    for inp in node.input:
        inits = find_initializers(model, inp)
        if len(inits) > 0:
            for value in inits:
                if value not in const_init:
                    const_init.append(value)
    return const_init


def remove_node_and_init_by_indexs(model, inodes, inints):
    inodes = sorted(inodes, reverse=True)
    inints = sorted(inints, reverse=True)
    for i in inodes:
        del model.graph.node[i]

    for i in inints:
        del model.graph.initializer[i]


def convert_any_to_python_primary_type(x: Union[np.ndarray, int, float, list, str], accept_none: bool=False) -> Union[int, float, list, str, None]:
    if x is None and accept_none: return None
    if x is None and not accept_none: raise ValueError('Trying to convert an empty value.')
    if isinstance(x, list) or isinstance(x, tuple): return list(x)
    elif isinstance(x, int) or isinstance(x, float): return x
    elif isinstance(x, np.ndarray):
        if x.size == 0 and accept_none: return None
        if x.size == 0 and not accept_none: raise ValueError('Trying to convert an empty value.')
        if x.size == 1: return x.reshape((1, )).tolist()[0]
        if x.size  > 1: return x.tolist()
    elif isinstance(x, str):
        return x
    else:
        raise TypeError(f'input value {x}({type(x)}) can not be converted as python primary type.')
