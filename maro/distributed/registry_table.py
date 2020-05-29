
# subconstraint = 'source:type:num'

class Constraint():
    def __init__(self, constraint: str (subconstriant or (subconstriant, subconstraint, operation))):
         # decode constraint
        self._subconstraint_dict = {} # {subconstraint: msg list}
        self._logic_relation = [[subconstraint, subconstraint, ...], [subconstraint, subconstraint, ...]]
        constraint = constraint.replace(' ', '')

    def _constraint_type(self, constraint):
        cparen_position = constraint.rfind(')')
        if cparen_position == -1:
            self._check_subconstraint_expression(constraint)
            self._subconstraint_dict[constraint] = []
            self._logic_relation.append([constraint])
        else:
            oparen_position = constraint[:cparen_position].find('(')
            self._constraint_decomposer(constraint[oparen_position+1:cparen_position])

    def _check_subconstraint_expression(self, sub_constraint):
        decomposer = sub_constraint.split(':')
        assert(len(decomposer)==3)
        # number must be int or percentage(*%)
        if decomposer[-1][-1] == '%':
            decomposer[-1] = decomposer[-1][:-1]
        
        try:
            number = int(decomposer[-1])
        except:
            print(f"number must be int or percentage with % in the end.")
    
    def _constraint_decomposer(self, constraint):
        operation_list = ['&&', '||', 'AND', 'OR']
        # clear constraint
        constraint = constraint.replace('(', '')
        constraint = constraint.replace(')', '')

        constraint_decomposer = constraint.split(',')
        op_index = []
        # check expression
        for idx, component in enumerate(constraint_decomposer):
            if component in operation_list:
                op_index.append(idx)
            else:
                self._check_subconstraint_expression(component)
        
        used_component_index = []
        for op_i in op_index:
            used_component_index.append(op_i)
            combined_constraint = False
            curr_index = op_i - 1
            while curr_index in used_component_index:
                curr_index -= 1
                combined_constraint = True
            if combined_constraint:
                self._logic_update(constraint_decomposer[curr_index], constraint_decomposer[op_i])
                used_component_index.append(curr_index)
            else:
                used_component_index.append(curr_index)
                used_component_index.append(curr_index)


    def _check_constraint_satisfied(self):
        # check if self satisfied

    def constraint_satisfied(self, message):
        # check message satisfied any sub-constaint in this Constriant


class RegisterTable():
    def __init__(self):
        self._handler_constraint_dict = {handler_fn: constraint class}        

    def add_constraint(self, constraint: class, handler_fn: callable):
        self._msg_request_handle_dict[handler_fn] = constraint

    def message_trigger(self, message):
        satisfied_handler_fn = []

        for handler_fn, constraint in self._handler_constraint_dict.items():
            msg_lst = constraint.constraint_satisfied(message)

            if msg_lst:    
                satisfied_handler_fn.append((handler_fn, msg_lst))
        
        return satisfied_handler_fn

    def handler_fn(self):
        return self._handler_constraint_dict.keys()


if __name__ == '__main__':
    n = RegisterTable()
    n.add_constraint([(env1, 'experience', 1), (env2, 'checkout', 1)], training)
    n.add_constraint((env1, 'experience', 1), training)