
class RegisterTable():
    def __init__(self):
        self._msg_request_handle_dict = {}
                                        {idx: {HandlerKey.REMAIN: {(env, type): num},
                                               HandlerKey.OPERATION: 'AND'/'OR'
                                               HandlerKey.MSG_LST: [],
                                               HandlerKey.HANDLER_FN: ..}}
        self._message_cache = []

    def add_constraint(self, constraint: tuple(env, msg_type, num) or list of tuple, 
                        handler_fn: callable, operation: AND or OR, only available on list constraint):
        constraint_index = self._generate_dict_index(constraint)
        self._msg_request_handle_dict[constraint_index] = ...

    def _generate_dict_index(self, constraint):
        return string -> "env+type+num ..."

    def _init_msg_request(self, constraint_index, handler_fn, operation):
        ...

    def _check_msg_satisfied(self, constraint_index):
        target_constraint = self._msg_request_handle_dict[constraint_index]

        if target_constraint[HandlerKey.OPERATION] == 'AND':
            for key, num in target_constraint[HandlerKey.REMAIN].items():
                if num != 0:
                    return False
            return True
        elif target_constraint[HandlerKey.OPERATION] == 'OR':
            for key, num in target_constraint[HandlerKey.REMAIN].items():
                if num == 0:
                    return True
            return False

    def _msg_in_constraint(self, msg_source, msg_type):
        idx = str(msg_source) + str(msg_type)
        satisfied_constraint = []

        for constriant_idx in self._msg_request_handle_dict.keys():
            if idx in constriant_idx:
                satisfied_constraint.append((constraint_idx, (msg_source, msg_type)))
            # other situation 
            idx = str(source) + str(ConstraintType.ANYTYPE)

        return satisfied_constraint

    def _message_trigger(self, message):
        satisfied_handler_fn = []
        satisfied_constraint = self._msg_in_constraint(message.source, message.type)
        message_spend = False

        for constraint_index, single_constraint in satisfied_constraint:
            remain_num = self._msg_request_handle_dict[constraint_index][HandlerKey.REMAIN][single_constraint]
            if remain_num != 0:
                self._msg_request_handle_dict[constraint_index][HandlerKey.REMAIN][single_constraint] -= 1
                self._msg_request_handle_dict[constraint_index][HandlerKey.MSG_LST].append(message)
                message_spend = True

            if self._check_msg_satisfied(constraint_index):
                handler_fn = self._msg_request_handle_dict[constraint_index][HandlerKey.HANDLER_FN]
                msg_lst = self._msg_request_handle_dict[constraint_index][HandlerKey.MSG_LST][:]
                satisfied_handler_fn.append((handler_fn, msg_lst))
                self._init_msg_request(constraint_index, handler_fn)
            
        if not message_spend:
            self._message_cache.append(message)

        return satisfied_handler_fn

    def message_trigger(self, message):
        satisfied_handler_fn = []

        pending_message = self._message_cache[:]
        pending_message.append(message)
        self._message_cache = []

        for msg in pending_message:
            satisfied_handler_fn.append(self._message_trigger(msg))
        
        return satisfied_handler_fn

    def handler_fn(self):
        return handler_fn_list


if __name__ == '__main__':
    n = RegisterTable()
    n.add_constraint([(env1, 'experience', 1), (env2, 'checkout', 1)], training)
    n.add_constraint((env1, 'experience', 1), training)