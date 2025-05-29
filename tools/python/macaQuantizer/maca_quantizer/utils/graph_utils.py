
from typing import List

from ppq.IR.search import SearchableGraph, Operation
from ppq import BaseGraph


def get_hybrid_path_op(graph: BaseGraph, op_path_names: List):
        """   
            get hybrid operator by operator name
        """
        search_engine = SearchableGraph(graph)

        match_nodes_list = []
        for path in op_path_names:
            assert len(path) == 2
            start_opname = path[0]
            end_opname = path[-1]

            paths = search_engine.path_matching(
                    sp_expr=lambda x: x.name == start_opname, 
                    rp_expr=lambda x, y: True, 
                    ep_expr=lambda x: x.name==end_opname, 
                    direction='down')

            path_ops = []
            for path in paths:
                for op in path:
                    path_ops.append(op.name)
            
            match_nodes_list.extend(path_ops)
        
        match_nodes_list = list(set(match_nodes_list))

        return match_nodes_list
                     
                  