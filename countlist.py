class CountList:
    def __init__(self, *args):
        self.values = [x for x in args]
        self.count = {}.fromkeys(range(len(self.values)), 0)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        self.count[index] += 1
        return self.values[index]

    def __setitem__(self, index, value):
        self.values[index] = value

    def __delitem__(self, index):
        del self.values[index]
        del self.count[index]

    