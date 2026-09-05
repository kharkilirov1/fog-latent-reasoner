"""Identity-blind local grammatical context, not absolute mention/name shortcuts."""
import torch
from torch import nn

class WindowRoleReader(nn.Module):
    def __init__(self,ref,d,width=32,radius=3,linear=False,ignore_prefix=False):
        super().__init__()
        self.radius=radius
        self.ignore_prefix=ignore_prefix
        self.mix=ref.HeadLayerMixer([0],0)
        self.project=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,width),nn.GELU())
        self.role=nn.Linear(2*radius*width,2,bias=False) if linear else nn.Sequential(nn.Linear(2*radius*width,96),nn.GELU(),nn.Linear(96,2))
    def forward(self,f,mask,mentions,valid):
        x=self.project(self.mix.ctx(f[:,:1].float()))
        x=x.masked_fill((~mask | mentions.any(1))[...,None],0.)
        n=x.size(1);chunks=[]
        if self.ignore_prefix:
            pos=torch.arange(n,device=x.device)
            first=pos[None].expand(x.size(0),-1).masked_fill(~mentions.any(1),n).min(-1).values
            x=x.masked_fill((pos[None]<first[:,None])[...,None],0.)
        for offset in range(-self.radius,self.radius+1):
            if offset==0:continue
            idx=torch.arange(n,device=x.device)+offset
            real=(idx>=0)&(idx<n)
            chunks.append(x[:,idx.clamp(0,n-1)]*real[None,:,None])
        local=torch.cat(chunks,-1)
        end=mentions & ~torch.nn.functional.pad(mentions[:,:,1:],(0,1))
        mm=end.float()
        slots=torch.einsum('bmt,btd->bmd',mm,local)/mm.sum(-1,keepdim=True).clamp_min(1)
        scores=self.role(slots)
        joint=scores[:,:,0,None]+scores[:,None,:,1]
        permitted=valid.clone();permitted[~valid.any(1),0]=True
        allowed=permitted[:,:,None]&permitted[:,None,:]
        diag=torch.eye(valid.size(-1),device=valid.device,dtype=torch.bool)
        allowed &= ~(diag[None] & (valid.sum(-1)>1)[:,None,None])
        return joint.masked_fill(~allowed,-1e4)

from model import SemanticReader,st_onehot

class TypedSemanticReader(SemanticReader):
    """Use an identity-blind binding head only when the PREDICTED opcode is BIND."""
    def __init__(self,ref,d,op_proto,relation_proto,bind_config=None):
        super().__init__(ref,d,op_proto,relation_proto)
        self.bind_config=bind_config or {'width':32,'radius':1,'linear':True,'ignore_prefix':False}
        self.bind_roles=WindowRoleReader(ref,d,**self.bind_config)
    def forward(self,f,mask,mentions,valid):
        op,rel,pair=super().forward(f,mask,mentions,valid)
        bind_pair=self.bind_roles(f,mask,mentions,valid)
        gate=st_onehot(op)[:,0,None,None]
        return op,rel,(1-gate)*pair+gate*bind_pair
