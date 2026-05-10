from __future__ import annotations
import time, torch, numpy as np

def rel_l2(a,b,eps=1e-8): return torch.linalg.vector_norm(a-b).item()/(torch.linalg.vector_norm(b).item()+eps)
def parameter_count(model): return sum(p.numel() for p in model.parameters())
def inference_time(model,batch,repeats=3):
    start=time.perf_counter()
    with torch.no_grad():
        for _ in range(repeats): model(batch['history'],batch.get('cond'),batch.get('static'))
    return (time.perf_counter()-start)/repeats

def oscillatory_metrics(pred_trace, ref_trace, dt=1.0):
    p=np.asarray(pred_trace).reshape(len(pred_trace),-1).mean(1); r=np.asarray(ref_trace).reshape(len(ref_trace),-1).mean(1)
    amp_err=abs((p.max()-p.min())-(r.max()-r.min()))/(abs(r.max()-r.min())+1e-8)
    pf=np.fft.rfftfreq(len(p),dt)[np.argmax(np.abs(np.fft.rfft(p-p.mean()))[1:])+1] if len(p)>2 else 0.0
    rf=np.fft.rfftfreq(len(r),dt)[np.argmax(np.abs(np.fft.rfft(r-r.mean()))[1:])+1] if len(r)>2 else 0.0
    corr=np.correlate(p-p.mean(),r-r.mean(),mode='full'); shift=int(np.argmax(corr)-(len(r)-1)); return {"amplitude_error":float(amp_err),"dominant_frequency_error":float(abs(pf-rf)/(abs(rf)+1e-8)),"phase_drift":float(shift*dt)}
