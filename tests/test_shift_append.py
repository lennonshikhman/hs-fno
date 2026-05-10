import torch
from hsno.data.history import shift_append

def test_shift_append_m1_1d():
    h=torch.arange(2*4*1*5.).view(2,4,1,5); p=torch.ones(2,1,5)*99; o=shift_append(h,p,1)
    assert torch.equal(o[:,:3],h[:,1:]); assert torch.equal(o[:,3],p)

def test_shift_append_m2_1d():
    h=torch.arange(1*5*1*3.).view(1,5,1,3); p=torch.ones(1,2,1,3)*7; o=shift_append(h,p,2)
    assert torch.equal(o[:,:3],h[:,2:]); assert torch.equal(o[:,3:],p)

def test_shift_append_m2_2d():
    h=torch.randn(2,5,1,4,3); p=torch.randn(2,2,1,4,3); o=shift_append(h,p,2)
    assert o.shape==h.shape; assert torch.equal(o[:,:3],h[:,2:]); assert torch.equal(o[:,3:],p)
