import numpy as np
from hsno.solvers.reaction_diffusion import ReactionDiffusionSolver
from hsno.solvers.epidemic import EpidemicDelaySolver
from hsno.solvers.neural_field import NeuralFieldSolver
from hsno.solvers.delayed_wave import DelayedWaveSolver
from hsno.solvers.distributed_memory import DistributedMemorySolver

def check(tr): assert np.isfinite(tr.fields).all() and tr.fields.shape[0] >= 2

def test_all_solver_smoke():
    base=dict(nx=12,dt=0.01,dt_save=0.05,total_time=0.1,history_steps=4,seed=1)
    check(ReactionDiffusionSolver().simulate({'D':0.005,'r':1.0,'tau':0.05},**base))
    check(EpidemicDelaySolver().simulate({'D':0.005,'beta':1.0,'gamma':0.5,'tau':0.05},**base))
    check(NeuralFieldSolver().simulate({'tau0':0.02,'alpha':0.03,'gain':1.0,'width':0.1},**base))
    check(DelayedWaveSolver().simulate({'c':0.2,'tau':0.05,'alpha':1.0,'beta':0.1},**base))
    check(DistributedMemorySolver().simulate({'nu':0.005,'a1':0.5,'a2':0.0,'tau':0.05},**base))

def test_d2_resolution_transfer_solver_smoke():
    base=dict(nx=8,ny=6,dim=2,dt=0.005,dt_save=0.025,total_time=0.05,history_steps=4,seed=2)
    check(ReactionDiffusionSolver().simulate({'D':0.002,'r':0.8,'tau':0.04},**base))
    check(DistributedMemorySolver().simulate({'nu':0.002,'a1':0.3,'a2':0.0,'tau':0.04},**base))
