from maro.backends.frame import NodeAttribute, NodeBase, node


@node("account")
class Account(NodeBase):
    """Account node definition in frame, used to maintain cash changing."""
    remaining_cash = NodeAttribute("f")
    assets_value = NodeAttribute("f")

    def set_init_state(self, init_cash: float):
        """Set initial state of the account.

        Args:
            init_cash (float): Initial cash in the account.
        """
        self._init_cash = init_cash
        self.remaining_cash = self._init_cash
        self.assets_value = 0

    def reset(self):
        self.remaining_cash = self._init_cash
        self.assets_value = 0
