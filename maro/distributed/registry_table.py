# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import itertools
import numpy as np

from enum import Enum
from maro.distributed.message import Message


class Operation(Enum):
    AND = 0
    OR = 1


class LogicTree():
    def __init__(self, value=None, parent=None, child=None):
        self.value = value
        self.child = child if child else []

    def show(self):
        print(self.value)
        queue = self.child[:]
        idx = 0
        while idx < len(queue):
            curr_q = queue[idx]
            if isinstance(curr_q, LogicTree):
                for item in curr_q.child:
                    queue.append(item)
                print(curr_q.value)
            else:
                print(curr_q)
            idx += 1
        

class Constraint():
    def __init__(self, constraint: str, peer_list=None):
        self._subconstraint_dict = {} # {subconstraint: msg list}
        self._logic_tree = LogicTree()
        self._peer_list = peer_list

        # regularize constraint input
        constraint = constraint.replace(' ', '')
        self._init_constraint(constraint)

    def _init_constraint(self, constraint):
        cparen_position = constraint.rfind(',')
        if cparen_position == -1:
            self._check_subconstraint_expression(constraint)
            self._subconstraint_dict[constraint] = []
            self._logic_tree.value = constraint
        else:
            self._constraint_decomposer(constraint)

    def _check_subconstraint_expression(self, sub_constraint):
        decomposer = sub_constraint.split(':')
        assert(len(decomposer)==3)
        # number must be int or percentage(*%)
        if decomposer[-1][-1] == '%':
            decomposer[-1] = decomposer[-1][:-1]
        
        try:
            number = int(decomposer[-1])
        except:
            raise ValueError(f"number in subconstraint must be int or percentage with % in the end.")
    
    def _logic_tree_generator(self, constraint, childtree=None):
        operation_and_list = ['&&', 'AND']
        operation_or_list = ['||', 'OR']
        temp_logictree = LogicTree()
        
        decomposer = constraint.split(',')
        for subc in decomposer:
            if subc in operation_and_list:
                temp_logictree.value = Operation.AND
            elif subc in operation_or_list:
                temp_logictree.value = Operation.OR
            elif subc != '#'*len(subc):
                self._check_subconstraint_expression(subc)
                self._subconstraint_dict[subc] = []
                temp_logictree.child.append(subc)
        
        if childtree:
            for tree in childtree:
                temp_logictree.child.append(tree)

        return temp_logictree
    
    def _constraint_decomposer(self, constraint):
        tree_idx_dict = {}
        while True:
            lparen_idx = constraint.find(')')
            rparen_idx = constraint[:lparen_idx].rfind('(')
            temp_tree_list = []
            for tree_idx in list(tree_idx_dict.keys()):
                if tree_idx > rparen_idx:
                    temp_tree_list.append(tree_idx_dict[tree_idx])
                    del tree_idx_dict[tree_idx]
            
            if rparen_idx == 0:
                self._logic_tree = self._logic_tree_generator(constraint[rparen_idx+1:lparen_idx], temp_tree_list)
                break
            else:
                newtree = self._logic_tree_generator(constraint[rparen_idx+1:lparen_idx], temp_tree_list)

            tree_idx_dict[rparen_idx] = newtree
            # replace used constraint by #
            constraint = constraint.replace(constraint[rparen_idx:lparen_idx+1], '#'*(lparen_idx+1-rparen_idx))

    def _get_target_component_number(self, source):
        target_source_group_number = 0
        for peer in self._peer_list:
            if source in peer:
                target_source_group_number += 1
        
        if target_source_group_number == 0:
            raise ValueError(f"There is no target source in peer list!")

        return target_source_group_number

    def _satisfied_subconstraint(self, subconstraint):
        source, _, num = subconstraint.split(':')
        if num[-1] == '%':
            component_group_num = self._get_target_component_number(source)
            num = np.floor(int(num[:-1])*component_group_num/100)

        if int(num) <= len(self._subconstraint_dict[subconstraint]):
            return [subconstraint]
        
        return []

    def _satisfied_constraint(self, logictree):
        operation = logictree.value
        result = []
        for subc in logictree.child:
            if isinstance(subc, LogicTree):
                result.append(self._satisfied_constraint(subc))
            else:
                result.append(self._satisfied_subconstraint(subc))

        # remove [] in result
        for r in result:
            if operation == Operation.AND and not r:
                return []
            if operation == Operation.OR and r:
                return r

        flatten_result = list(itertools.chain.from_iterable(result))
        return flatten_result

    def _check_constraint_satisfied(self):
        # check if constraint satisfied
        if isinstance(self._logic_tree.value, Operation):
            return self._satisfied_constraint(self._logic_tree)
        else:
            return self._satisfied_subconstraint(self._logic_tree.value)

    def push_msg(self, message):
        # check if message satisfied any subconstraint and put msg in target msg_list
        message_source, message_type = message.source, message.type
        for subconstraint in self._subconstraint_dict.keys():
            source, s_type, _ = subconstraint.split(':')
            source_match, type_match = False, False
            if source == message_source or source == '*' or source in message_source:
                source_match = True
            if s_type == message_type or s_type == '*':
                type_match = True

            if source_match and type_match:
                self._subconstraint_dict[subconstraint].append(message)

    def pull_msg(self):
        total_msg_list = []
        satisfied_subconstraint_list = self._check_constraint_satisfied()

        if satisfied_subconstraint_list:
            if isinstance(satisfied_subconstraint_list, list):
                for subc in satisfied_subconstraint_list:
                    source, _, num = subc.split(':')
                    if num[-1] == '%':
                        component_group_num = self._get_target_component_number(source)
                        num = np.floor(int(num[:-1])*component_group_num/100)
                    num = int(num)
                    total_msg_list.append(self._subconstraint_dict[subc][:num])
                    del self._subconstraint_dict[subc][:num]
            else:
                num = int(satisfied_subconstraint_list.split(':')[2])
                total_msg_list.append(self._subconstraint_dict[satisfied_subconstraint_list][:num])
                del self._subconstraint_dict[satisfied_subconstraint_list][:num]
        
        # flatten
            total_msg_list = list(itertools.chain.from_iterable(total_msg_list))

        return total_msg_list

    def auto_satisfied(self, message):
        # check message satisfied any sub-constaint in this Constriant
        self.push_msg(message)
        msg_list = self.pull_msg()

        return msg_list

    def update_peers(self, peers):
        self._peer_list = peers


class RegisterTable():
    def __init__(self, peer_list):
        self._handler_constraint_dict = {}
        self._peer_list = peer_list
    def register_constraint(self, constraint: str, handler_fn: callable):
        constraint_class = Constraint(constraint, self._peer_list)
        self._handler_constraint_dict[handler_fn] = constraint_class
        # constraint_class._logic_tree.show()

    def push(self, message):
        satisfied_handler_fn = []

        for handler_fn, constraint in self._handler_constraint_dict.items():
            msg_lst = constraint.auto_satisfied(message)

            if msg_lst:    
                satisfied_handler_fn.append((handler_fn, msg_lst))
        
        return satisfied_handler_fn

    def update_peers(self, peers):
        for constraint in self._handler_constraint_dict.values():
            constraint.update_peers(peers)


if __name__ == '__main__':
    # subconstraints
    subc1 = 'env1:experience:1'
    subc2 = 'env2:experience:1'
    subc3 = 'env3:experience:1'
    subc4 = 'env4:experience:1'
    subc5 = 'env5:experience:1'
    subc6 = 'env:experience:40%'
    subc7 = '*:experience:1'
    subc8 = 'env5:*:1'
    subc9 = 'env:experience:3'
    
    
    # constraints
    cons1 = f"({subc1}, {subc2}, {subc3}, AND)"
    cons2 = f"({subc1}, {subc2}, {subc3}, OR)"
    cons3 = f"(({subc1}, {subc2}, OR), {subc4}, AND)"
    cons4 = f"(({subc3}, {subc4}, AND), ({subc1}, {subc2}, AND), {subc5}, AND)"
    cons5 = f"((({subc1}, {subc2}, OR), {subc6}, AND), {subc4}, AND)"

    # messages
    message1 = Message(type='experience', source='env1', destination='l1')
    message2 = Message(type='experience', source='env2', destination='l1')
    message3 = Message(type='experience', source='env3', destination='l1')
    message4 = Message(type='experience', source='env4', destination='l1')
    message5 = Message(type='experience', source='env5', destination='l1')
    message6 = Message(type='check_out', source='env5', destination='l1')

    message_pool = [message1,message2,message3,message4,message5,message6]
    
    # registerTabel
    n = RegisterTable(['env1', 'env2', 'env3', 'env4', 'env5'])
    n.register_constraint(subc1, 'testsubfn1')
    n.register_constraint(subc6, 'testsubfn6')
    n.register_constraint(subc7, 'testsubfn7')
    n.register_constraint(subc8, 'testsubfn8')
    n.register_constraint(cons1, 'testconsfn1')
    n.register_constraint(cons2, 'testconsfn2')
    n.register_constraint(cons3, 'testconsfn3')
    n.register_constraint(cons4, 'testconsfn4')
    n.register_constraint(cons5, 'testconsfn5')

    for msg in message_pool:
        print(n.push(msg))

    # * --> anysource/ anyType
    # env -> any env source
    # regular -> env_1:training:3
    # percentage -> env_1:training:80%