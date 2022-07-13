import cmx_cffi
import numpy as np
import cffi

ffi = cffi.FFI()

sizelow=2
sizehigh=5

def to_ctypes(matrix):
    lenx, leny = matrix.shape
    flattened = matrix.flatten()
    ptr = ffi.cast('float*', flattened.ctypes.data)
    return ptr, lenx, leny, flattened

def testmult():
    d1=np.random.randint(sizelow, sizehigh)
    d2=np.random.randint(sizelow, sizehigh)
    d3=np.random.randint(sizelow, sizehigh)

    a = np.random.rand(d1,d2).astype(np.float32)
    aptr, ax, ay, af = to_ctypes(a)

    b = np.random.rand(d2,d3).astype(np.float32)
    bptr, bx, by, bf = to_ctypes(b)

    res = np.zeros((ax, by), dtype=np.float32)
    resptr, rx, ry, resf = to_ctypes(res)

    cmx_cffi.lib.matmul(aptr, ax, ay, bptr, bx, by, resptr)
    res = resf.reshape((rx, ry))

    n = np.matmul(a,b)
    diff = res - n
    return sum(sum(diff)), ax, ay, bx, by

def testmulttrans():
    d1=np.random.randint(sizelow, sizehigh)
    d2=np.random.randint(sizelow, sizehigh)
    d3=np.random.randint(sizelow, sizehigh)

    a = np.random.rand(d1,d2).astype(np.float32)
    aptr, ax, ay, af = to_ctypes(a)

    b = np.random.rand(d3,d2).astype(np.float32)
    bptr, bx, by, bf = to_ctypes(b)

    res = np.zeros((ax, bx), dtype=np.float32)
    resptr, rx, ry, resf = to_ctypes(res)

    cmx_cffi.lib.matmul_trans(aptr, ax, ay, bptr, bx, by, resptr)
    res = resf.reshape((rx, ry))

    n = np.matmul(a,b.T)
    diff = res - n
    return sum(sum(diff)), ax, ay, bx, by

def testrownorm():
    a = np.asarray([[1,2,3],[4,5,6]], dtype=np.float32)
    aptr, ax, ay, af = to_ctypes(a)

    cmx_cffi.lib.rownorm(aptr, ax, ay)

    print(af)



if __name__ == "__main__":
    #testrownorm()
    print(testmulttrans())
#    bad=0
#    tot=100
#
#    gs={}
#    bs={}
#
#    for _ in range(tot):
#        e, ax, ay, bx, by = testmult()
#
#        id = (ax, ay, bx, by)
#
#        if abs(e) > 0.01:
#            bad = bad + 1
#            try:
#                bs[id] += 1
#            except KeyError:
#                bs[id] = 1
#        else:
#            try:
#                gs[id] += 1
#            except KeyError:
#                gs[id] = 1
#
#    for i in range(sizelow, sizehigh):
#        for j in range(sizelow, sizehigh):
#            for k in range(sizelow, sizehigh):
#                for l in range(sizelow, sizehigh):
#
#                    if j != k:
#                        continue
#
#                    id = (i, j, k, l)
#
#                    try:
#                        g = gs[id]
#                    except KeyError:
#                        g = 0
#
#                    try:
#                        b = bs[id]
#                    except KeyError:
#                        b = 0
#
#                    #print(i, j, "*", k, l, ":", "%6d" % g, "%6d" % b)
#
#    print("Bad: ", bad, "/", tot, "=", "%5.2f" % (bad/tot*100), "%")
#
#