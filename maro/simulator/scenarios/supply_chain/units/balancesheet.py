
class BalanceSheet:
    def __init__(self, profit: int = 0, loss: int = 0):
        self.profit = profit
        self.loss = loss

    def total(self) -> int:
        return self.profit + self.loss

    def __add__(self, other):
        return BalanceSheet(self.profit + other.profit, self.loss + other.loss)

    def __sub__(self, other):
        return BalanceSheet(self.profit - other.profit, self.loss - other.loss)

    def __repr__(self):
        return f"{round(self.profit+self.loss, 0)} ({round(self.profit, 0)} {round(self.loss, 0)})"

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)
