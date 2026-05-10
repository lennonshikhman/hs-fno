from __future__ import annotations
import warnings
import torch
import torch.nn as nn
from hsno.data.history import shift_append
from hsno.utils.naming import canonical_model_name
from .hsno import FutureSliceOperator, HSNO, HSNOFull, with_step_condition
from .hsfno import HistorySpaceFNO
from .convlstm import ConvLSTMOperator
from .temporal_transformer import TemporalTransformerOperator
from .temporal_unet import TemporalUNetOperator


class CurrentStateNO(nn.Module):
    def __init__(self, history_steps, channels, cond_dim=0, static_channels=0, step_slices=1, **kw):
        super().__init__(); self.op=FutureSliceOperator(1,channels,cond_dim,static_channels,step_slices,**kw); self.step_slices=step_slices
    def forward(self, history, cond=None, static=None, delta_t=None, cond_mean=None, cond_std=None):
        cond=with_step_condition(cond,delta_t,cond_mean,cond_std); future=self.op(history[:,-1:],cond,static); return shift_append(history,future,self.step_slices)
    def predict_future(self, history, cond=None, static=None, delta_t=None, cond_mean=None, cond_std=None):
        cond=with_step_condition(cond,delta_t,cond_mean,cond_std); return self.op(history[:,-1:],cond,static)


class LagStackNO(nn.Module):
    def __init__(self, history_steps, channels, lags=3, cond_dim=0, static_channels=0, step_slices=1, **kw):
        super().__init__(); self.lags=min(lags,history_steps); self.op=FutureSliceOperator(self.lags,channels,cond_dim,static_channels,step_slices,**kw); self.step_slices=step_slices
    def forward(self, history, cond=None, static=None, delta_t=None, cond_mean=None, cond_std=None):
        cond=with_step_condition(cond,delta_t,cond_mean,cond_std); future=self.op(history[:,-self.lags:],cond,static); return shift_append(history,future,self.step_slices)
    def predict_future(self, history, cond=None, static=None, delta_t=None, cond_mean=None, cond_std=None):
        cond=with_step_condition(cond,delta_t,cond_mean,cond_std); return self.op(history[:,-self.lags:],cond,static)


class HistoryToHistoryNO(nn.Module):
    def __init__(self, history_steps, channels, cond_dim=0, static_channels=0, step_slices=1, backbone="conv", width=32, depth=3, dim=1, **kw):
        super().__init__(); self.history_steps=history_steps; self.channels=channels; self.step_slices=step_slices
        self.op=FutureSliceOperator(history_steps,channels,cond_dim,static_channels,history_steps,backbone,width,depth,dim,kw.get("coordinate_embedding",False),kw.get("film",False))
    def forward(self, history, cond=None, static=None, delta_t=None, cond_mean=None, cond_std=None):
        cond=with_step_condition(cond,delta_t,cond_mean,cond_std); return self.op(history,cond,static)
    def predict_future(self, history, cond=None, static=None, delta_t=None, cond_mean=None, cond_std=None): return self.forward(history,cond,static,delta_t,cond_mean,cond_std)[:,-self.step_slices:]


class SequenceNO(nn.Module):
    def __init__(self, kind, history_steps, channels, cond_dim=0, static_channels=0, step_slices=1, width=32, depth=3, dim=1, backbone="conv", **kw):
        super().__init__(); self.step_slices=step_slices
        if kind=="convlstm":
            self.seq = ConvLSTMOperator(channels,cond_dim,step_slices,width,dim)
        elif kind=="temporal_unet":
            self.seq = TemporalUNetOperator(channels,cond_dim,step_slices,width,max(1,depth))
        else:
            self.seq = TemporalTransformerOperator(channels,cond_dim,step_slices,width)
        self.readout = FutureSliceOperator(step_slices, channels, cond_dim, static_channels, step_slices, backbone, width, depth, dim, kw.get("coordinate_embedding",False), kw.get("film",False))
    def forward(self, history, cond=None, static=None, delta_t=None, cond_mean=None, cond_std=None):
        future = self.predict_future(history, cond, static, delta_t, cond_mean, cond_std)
        return shift_append(history, future, self.step_slices)
    def predict_future(self, history, cond=None, static=None, delta_t=None, cond_mean=None, cond_std=None):
        cond=with_step_condition(cond,delta_t,cond_mean,cond_std)
        base = self.seq(history, cond if cond is not None else torch.empty(history.shape[0],0,device=history.device))
        return self.readout(base, cond, static)


def _with(kwargs, **updates):
    out = dict(kwargs); out.update(updates); return out


def build_model(name, **kwargs):
    original = name
    name = canonical_model_name(name, warn=original != canonical_model_name(original))
    if original != name:
        warnings.warn(f"Legacy model name '{original}' remapped to '{name}'.", DeprecationWarning, stacklevel=2)
    if name=="current_state": return CurrentStateNO(**kwargs)
    if name=="lag_stack": return LagStackNO(**kwargs)
    if name=="history2history": return HistoryToHistoryNO(**kwargs)
    if name in {"hs_fno_no_shift"}: return HistoryToHistoryNO(**_with(kwargs, backbone="fno"))
    if name in {"hsno_unet_no_shift", "hsno_no_shift"}: return HistoryToHistoryNO(**_with(kwargs, backbone="conv"))
    if name=="convlstm": return SequenceNO("convlstm",**kwargs)
    if name=="temporal_transformer": return SequenceNO("transformer",**kwargs)
    if name=="temporal_unet": return SequenceNO("temporal_unet",**kwargs)
    if name in {"hs_fno", "hs_fno_coord_conditioning", "hs_fno_film_conditioning", "hs_fno_no_delay_conditioning", "hs_fno_history_steps_4", "hs_fno_history_steps_8", "hs_fno_history_steps_12", "hs_fno_history_resolution_4", "hs_fno_history_resolution_8", "hs_fno_history_resolution_12", "hs_fno_per_delay_low", "hs_fno_per_delay_high"}:
        return HistorySpaceFNO(**kwargs)
    if name=="hs_fno_rollout_semiflow": return HistorySpaceFNO(**_with(kwargs, append_mode="recursive"))
    if name=="hs_transformer": return HSNO(**_with(kwargs, backbone="transformer"))
    if name=="hsno_unet": return HSNO(**_with(kwargs, backbone="conv"))
    if name=="hsno_unet_rollout_semiflow": return HSNOFull(**_with(kwargs, backbone="conv"))
    if name=="hsno_full": return HSNOFull(**kwargs)
    if name=="hsno": return HSNO(**kwargs)
    raise ValueError(name)
