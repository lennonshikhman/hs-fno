import numpy as np
from hsno.data.history import make_tau_history_windows


def test_tau_history_windows_use_delay_grid():
    times=np.array([-0.2,-0.1,0.0,0.1,0.2],dtype=np.float32)
    fields=times[:,None,None].astype(np.float32)
    xs,ys,fut,roll=make_tau_history_windows(fields,times,tau=0.2,history_steps=3,step_slices=1,delta_t=0.1,rollout_steps=1)
    assert np.allclose(xs[0][:,0,0],[-0.2,-0.1,0.0])
    assert np.allclose(ys[0][:,0,0],[-0.1,0.0,0.1])
    assert np.allclose(fut[0][:,0,0],[0.1])
    assert np.allclose(roll[0][0,:,0,0],[-0.1,0.0,0.1])
