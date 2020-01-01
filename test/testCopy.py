import copy

class C():
    def __init__(self):
        self.x = [1]
        self.y = [2]

# It copies.
c = C()
d = copy.copy(c)
d.x = [3]
assert c.x == [1]
assert d.x == [3]

# It's shallow.
c = C()
d = copy.copy(c)
d.x[0] = 3
assert c.x == [3]
assert d.x == [3]