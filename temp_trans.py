class CDesc:
    def __init__(self, value = 26.0):
        self.value = float(value)

    def __get__(self, instance, owner):
        return self.value
        
    def __set__(self, instance, value):
        self.value = float(value)


class FDesc:
    def __get__(self, instance, owner):
        return instance.c * 1.8 + 32
        
    def __set__(self, instance, value):
        instance.c = (float(value) - 32) / 1.8

class Temp:
    c = CDesc()
    f = FDesc()
