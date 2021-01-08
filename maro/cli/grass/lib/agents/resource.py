class BasicResource:
    def __init__(self, cpu: float, memory: float, gpu: float):
        self.cpu = cpu
        self.memory = memory
        self.gpu = gpu

    def __cmp__(self, other):
        if self.cpu == other.cpu and self.memory == other.memory and self.gpu == other.gpu:
            return 0
        elif self.cpu >= other.cpu and self.memory >= other.memory and self.gpu >= other.gpu:
            return 1
        else:
            return -1

    def __lt__(self, other):
        return self.__cmp__(other=other) == -1

    def __le__(self, other):
        return self.__cmp__(other=other) <= 0

    def __eq__(self, other):
        return self.__cmp__(other=other) == 0

    def __ne__(self, other):
        return self.__cmp__(other=other) != 0

    def __gt__(self, other):
        return self.__cmp__(other=other) == 1

    def __ge__(self, other):
        return self.__cmp__(other=other) >= 0

    def __add__(self, other):
        self.cpu += other.cpu
        self.memory += other.memory
        self.gpu += other.gpu

    def __radd__(self, other):
        self.__add__(other)

    def __iadd__(self, other):
        self.__add__(other)

    def __sub__(self, other):
        self.cpu -= other.cpu
        self.memory -= other.memory
        self.gpu -= other.gpu

    def __rsub__(self, other):
        self.__sub__(other)

    def __isub__(self, other):
        self.__sub__(other)

    def __call__(self):
        return {
            "cpu": self.cpu,
            "memory": self.memory,
            "gpu": self.gpu
        }


class ContainerResource(BasicResource):
    def __init__(self, container_name: str, cpu: float, memory: float, gpu: float):
        super().__init__(cpu=cpu, memory=memory, gpu=gpu)
        self.container_name = container_name


class NodeResource(BasicResource):
    def __init__(self, node_name: str, cpu: float, memory: float, gpu: float):
        super().__init__(cpu=cpu, memory=memory, gpu=gpu)
        self.node_name = node_name
