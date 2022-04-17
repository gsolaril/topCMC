import sys
class ProgBar:
    def __init__(self, items: list, width: int = 40, verbose: str = ""):

        self.__verbose, self.__width, self.__prog = verbose, width, 0
        if not isinstance(items, int):
            self.__length = len(items)
            self.__items = list(items) + ["OK"]
            chars = max(map(len, self.__items))
            add = lambda x: x.ljust(1 + chars - len(x))
            self.__items = list(map(add, self.__items))
        else: self.__length, self.__items = items, list()
        self.show()

    def reset(self): self.__prog = 0 ; print("\n")

    def show(self, step: int = 1):

        if not self.__items: item = self.__prog
        else: item = self.__items[self.__prog]
        if self.__verbose: verbose = "%s: %s" % (self.__verbose, item)
        if not self.__items: verbose += " / " + str(self.__length)
        track = int(self.__width * self.__prog / self.__length)
        bar = "=" * int(track) + " " * int(self.__width - track)
        pc = 100 * track // self.__width
        sys.stdout.write("\r[%s] %d%% | %s" % (bar, pc, verbose))
        if (self.__prog > self.__length): self.reset()
        sys.stdout.flush() ; self.__prog += step