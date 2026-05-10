import numpy as np
from hsno.solvers.delay_buffer import DelayBuffer

def test_lookup_exact_and_interp():
    b=DelayBuffer([0.0,1.0,2.0],[np.array([0.0]),np.array([2.0]),np.array([4.0])])
    assert np.allclose(b.lookup(1.0),[2.0]); assert np.allclose(b.lookup(0.5),[1.0]); assert np.allclose(b.lookup(-1),[0.0]); assert np.allclose(b.lookup(5),[4.0])

def test_pairwise_source_lookup_vectorized():
    b=DelayBuffer([0.0,1.0],[np.array([[0.0,2.0,4.0]]),np.array([[10.0,12.0,14.0]])])
    q=np.array([[0.5,0.5,0.5],[1.0,0.0,0.25]])
    src=np.array([[0,1,2],[2,1,0]])
    out=b.lookup_pairwise_sources(q,src)
    assert np.allclose(out,[[5.0,7.0,9.0],[14.0,2.0,2.5]])
