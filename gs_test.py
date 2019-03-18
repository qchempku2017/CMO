#!/usr/bin/env python

from gs_tools import GSsemigrand
import json as js

d=js.load(open('LMMOF_gs.mson'))
gs_skt = GSsemigrand.from_dict(d)
gs_skt.set_transmat([[1.0,1.0,-1.0],[1.0,-1.0,1.0],[-1.0,1.0,1.0]])
gs_skt.solve()
