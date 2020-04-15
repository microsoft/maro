class Demand:
    def __init__(self, demand: int):
        self.demand = demand

    def __repr__(self):
        return f"(Demand value: {self.demand})"

class Action:
    def __init__(self, restock: int):
        self.restock = restock

    def __repr__(self):
        return f"Action(restock: {self.restock})"