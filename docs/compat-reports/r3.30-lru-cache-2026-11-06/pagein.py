"""Samples \Memory\Pages Input/sec (hard page reads) once a second until FILE stop appears.
English counter names through PDH, since this host's Windows is localized.
Usage: pagein.py OUT.json STOPFILE"""
import ctypes as C, json, os, sys, time
pdh = C.WinDLL("pdh")
q = C.c_void_p(); c = C.c_void_p()
assert pdh.PdhOpenQueryW(None, None, C.byref(q)) == 0
assert pdh.PdhAddEnglishCounterW(q, "\Memory\Pages Input/sec", None, C.byref(c)) == 0
class V(C.Structure):
    _fields_ = [("status", C.c_uint32), ("pad", C.c_uint32), ("val", C.c_double)]
pdh.PdhCollectQueryData(q)
out = []
t0 = time.time()
while not os.path.exists(sys.argv[2]) and time.time() - t0 < 300:
    time.sleep(1)
    pdh.PdhCollectQueryData(q)
    v = V()
    pdh.PdhGetFormattedCounterValue(c, 0x200, None, C.byref(v))
    out.append(round(v.val))
json.dump(out, open(sys.argv[1], "w"))
