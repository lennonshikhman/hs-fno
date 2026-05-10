import torch
from hsno.models.baselines import build_model

def test_model_shapes():
    h=torch.randn(2,4,1,16); c=torch.randn(2,3)
    for name in ['current_state','lag_stack','history2history','hs_fno','hsno_unet','hs_transformer','convlstm','temporal_unet','temporal_transformer']:
        m=build_model(name,history_steps=4,channels=1,cond_dim=3,static_channels=0,step_slices=1,width=8,depth=1,dim=1)
        y=m(h,c,None); assert y.shape==h.shape
