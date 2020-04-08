class output_level:
    WHITE = '\033[0m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def output(color, string, end="\n"):
    print(color + string + output_level.ENDC, end=end)