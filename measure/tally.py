from sys import stdin

h = {}

for line in stdin:
    ss = line.split()
    if len(ss) == 2:
        m = int(ss[1])
        try:
            n = h[ss[0]]
            print(f"{ss[0]}\t{m-n}")
        except KeyError:
            pass
        h[ss[0]] = m
    else:
        print(line, end="")
