LIST_LEN = [[1,2,3], [1,2,3], [1,2], [1,2,3], [1,2,3,4], [1,2]]

def splitAndOneHot(l):
    y = int(l[0])
    x = []
    for i in range(len(l[1:])):
        for j in LIST_LEN[i]:
            if j == int(l[1:][i]):
                x.append(1)
            else:
                x.append(0)
    return x, y


