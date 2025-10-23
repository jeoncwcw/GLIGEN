from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

from ldm.util import prepare_tv_mask

# from ldm.modules.diffusionmodules.util import checkpoint, FourierEmbedder
from torch.utils import checkpoint

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


def resolve_tv_mask(tv_mask, n_visual_tokens, device):
    if tv_mask is None:
        return None

    if isinstance(tv_mask, dict):
        blob = tv_mask.get("blob")
        if blob is None:
            return None
        cache = tv_mask.setdefault("cache", {})
        size = int(math.sqrt(n_visual_tokens))
        key = size
        if key not in cache:
            cache[key] = prepare_tv_mask(blob, size, size)
        return cache[key].to(device)

    return tv_mask.to(device)


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)





class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, heads=8, dim_head=64, dropout=0):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads


        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(key_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(value_dim, inner_dim, bias=False)


        self.to_out = nn.Sequential( nn.Linear(inner_dim, query_dim), nn.Dropout(dropout) )


    def fill_inf_from_mask(self, sim, mask):
        if mask is not None:
            B,M = mask.shape
            mask = mask.unsqueeze(1).repeat(1,self.heads,1).reshape(B*self.heads,1,-1)
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(~mask, max_neg_value)
        return sim 


    def forward(self, x, key, value, mask=None):

        q = self.to_q(x)     # B*N*(H*C)
        k = self.to_k(key)   # B*M*(H*C)
        v = self.to_v(value) # B*M*(H*C)
   
        B, N, HC = q.shape 
        _, M, _ = key.shape
        H = self.heads
        C = HC // H 

        q = q.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C) # (B*H)*N*C
        k = k.view(B,M,H,C).permute(0,2,1,3).reshape(B*H,M,C) # (B*H)*M*C
        v = v.view(B,M,H,C).permute(0,2,1,3).reshape(B*H,M,C) # (B*H)*M*C

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale # (B*H)*N*M
        self.fill_inf_from_mask(sim, mask)
        attn = sim.softmax(dim=-1) # (B*H)*N*M

        out = torch.einsum('b i j, b j d -> b i d', attn, v) # (B*H)*N*C
        out = out.view(B,H,N,C).permute(0,2,1,3).reshape(B,N,(H*C)) # B*N*(H*C)

        return self.to_out(out)




class SelfAttention(nn.Module):
    def __init__(self, query_dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout) )

        # Flags used for self-attention sharing across different forward passes.
        self._record_kv = False
        self._record_detach = True
        self._cached_k = None
        self._cached_v = None
        self._kv_override = None

    def enable_kv_recording(self, detach=True):
        """Stores the computed key/value tensors during the next forward pass."""
        self._record_kv = True
        self._record_detach = detach
        self._cached_k = None
        self._cached_v = None

    def disable_kv_recording(self, clear_cache=False):
        self._record_kv = False
        if clear_cache:
            self.clear_kv_cache()

    def get_kv_cache(self):
        if self._cached_k is None or self._cached_v is None:
            return None
        return self._cached_k, self._cached_v

    def clear_kv_cache(self):
        self._cached_k = None
        self._cached_v = None

    def set_kv_override(self, cache, detach=True):
        if cache is None:
            self._kv_override = None
            return
        k, v = cache
        if detach:
            k = k.detach().clone()
            v = v.detach().clone()
        self._kv_override = (k, v)

    def clear_kv_override(self):
        self._kv_override = None
        self.clear_kv_cache()

    def forward(self, x, attn_mask=None):
        q = self.to_q(x) # B*N*(H*C)

        if self._kv_override is not None:
            k, v = self._kv_override
        else:
            k = self.to_k(x) # B*N*(H*C)
            v = self.to_v(x) # B*N*(H*C)
            if self._record_kv:
                if self._record_detach:
                    self._cached_k = k.detach().clone()
                    self._cached_v = v.detach().clone()
                else:
                    self._cached_k = k
                    self._cached_v = v
                self._record_kv = False

        B, N, HC = q.shape 
        H = self.heads
        C = HC // H 

        q = q.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C) # (B*H)*N*C
        k = k.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C) # (B*H)*N*C
        v = v.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C) # (B*H)*N*C

        sim = torch.einsum('b i c, b j c -> b i j', q, k) * self.scale  # (B*H)*N*N
        
        ################################
        # Self-attention에 attention mask 지원
        if attn_mask is not None:
            attn_mask = attn_mask.to(sim.device)
            if attn_mask.dtype != torch.bool:
                attn_mask = attn_mask.bool()
            attn_mask = attn_mask.unsqueeze(1).repeat(1, H, 1, 1).reshape(B*H, N, N)
            max_neg = -torch.finfo(sim.dtype).max
            sim.masked_fill_(~attn_mask, max_neg)
        ################################
        
        attn = sim.softmax(dim=-1) # (B*H)*N*N
        out = torch.einsum('b i j, b j c -> b i c', attn, v) # (B*H)*N*C
        out = out.view(B,H,N,C).permute(0,2,1,3).reshape(B,N,(H*C)) # B*N*(H*C)

        return self.to_out(out)



class GatedCrossAttentionDense(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, n_heads, d_head):
        super().__init__()
        
        self.attn = CrossAttention(query_dim=query_dim, key_dim=key_dim, value_dim=value_dim, heads=n_heads, dim_head=d_head) 
        self.ff = FeedForward(query_dim, glu=True)

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        self.register_parameter('alpha_attn', nn.Parameter(torch.tensor(0.)) )
        self.register_parameter('alpha_dense', nn.Parameter(torch.tensor(0.)) )

        # this can be useful: we can externally change magnitude of tanh(alpha)
        # for example, when it is set to 0, then the entire model is same as original one 
        self.scale = 1  

    def forward(self, x, objs):

        x = x + self.scale*torch.tanh(self.alpha_attn) * self.attn( self.norm1(x), objs, objs)  
        x = x + self.scale*torch.tanh(self.alpha_dense) * self.ff( self.norm2(x) ) 
        
        return x 


class GatedSelfAttentionDense(nn.Module):
    def __init__(self, query_dim, context_dim,  n_heads, d_head):
        super().__init__()
        
        # we need a linear projection since we need cat visual feature and obj feature
        self.linear = nn.Linear(context_dim, query_dim)

        self.attn = SelfAttention(query_dim=query_dim, heads=n_heads, dim_head=d_head)
        self.ff = FeedForward(query_dim, glu=True)

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        self.register_parameter('alpha_attn', nn.Parameter(torch.tensor(0.)) )
        self.register_parameter('alpha_dense', nn.Parameter(torch.tensor(0.)) )

        # this can be useful: we can externally change magnitude of tanh(alpha)
        # for example, when it is set to 0, then the entire model is same as original one 
        self.scale = 1  


    def forward(self, x, objs, tv_mask = None):
        # mask 지원하도록 코드 수정
        # x: (B, N_visual, D), objs: (B, N_text, C)
        B, N_visual, _ = x.shape
        objs = self.linear(objs)
        tokens = torch.cat([x,objs],dim=1)  # (B, N_visual + N_text, D)
        
        full_mask = None
        resolved_mask = resolve_tv_mask(tv_mask, N_visual, tokens.device)
        if resolved_mask is not None:
            N_text = tokens.shape[1] - N_visual
            if resolved_mask.shape[0] == B and resolved_mask.shape[1] == N_text and resolved_mask.shape[2] == N_visual:
                full_mask = torch.ones(B, N_visual + N_text, N_visual + N_text, dtype=torch.bool, device=tokens.device)
                full_mask[:, N_visual:, :N_visual] = resolved_mask.bool()
            else:
                full_mask = None
            
        out = self.attn(self.norm1(tokens), attn_mask=full_mask)
        out_vis = out[:, :N_visual, :]

        x = x + self.scale*torch.tanh(self.alpha_attn) * out_vis
        x = x + self.scale*torch.tanh(self.alpha_dense) * self.ff( self.norm2(x) )  
        
        return x 






class GatedSelfAttentionDense2(nn.Module):
    def __init__(self, query_dim, context_dim,  n_heads, d_head):
        super().__init__()
        
        # we need a linear projection since we need cat visual feature and obj feature
        self.linear = nn.Linear(context_dim, query_dim)

        self.attn = SelfAttention(query_dim=query_dim, heads=n_heads, dim_head=d_head)
        self.ff = FeedForward(query_dim, glu=True)

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        self.register_parameter('alpha_attn', nn.Parameter(torch.tensor(0.)) )
        self.register_parameter('alpha_dense', nn.Parameter(torch.tensor(0.)) )

        # this can be useful: we can externally change magnitude of tanh(alpha)
        # for example, when it is set to 0, then the entire model is same as original one 
        self.scale = 1  


    def forward(self, x, objs, tv_mask=None):

        B, N_visual, _ = x.shape
        B, N_ground, _ = objs.shape

        objs = self.linear(objs)
        
        # sanity check 
        size_v = math.sqrt(N_visual)
        size_g = math.sqrt(N_ground)
        assert int(size_v) == size_v, "Visual tokens must be square rootable"
        assert int(size_g) == size_g, "Grounding tokens must be square rootable"
        size_v = int(size_v)
        size_g = int(size_g)

        # tv_mask 적용
        tokens = torch.cat([x, objs], dim=1)

        full_mask = None
        resolved_mask = resolve_tv_mask(tv_mask, N_visual, tokens.device)
        if resolved_mask is not None:
            N_text = objs.shape[1]
            if resolved_mask.shape[0] == B and resolved_mask.shape[1] == N_text and resolved_mask.shape[2] == N_visual:
                full_mask = torch.ones(B, N_visual + N_text, N_visual + N_text, dtype=torch.bool, device=tokens.device)
                full_mask[:, N_visual:, :N_visual] = resolved_mask.bool()
            else:
                full_mask = None

        # select grounding token and resize it to visual token size as residual 
        out = self.attn(self.norm1(tokens), attn_mask=full_mask)[:, N_visual:, :]  # objs 위치의 출력
        out = out.permute(0,2,1).reshape( B,-1,size_g,size_g )
        out = torch.nn.functional.interpolate(out, (size_v,size_v), mode='bicubic')
        residual = out.reshape(B,-1,N_visual).permute(0,2,1)
        
        # add residual to visual feature 
        x = x + self.scale*torch.tanh(self.alpha_attn) * residual
        x = x + self.scale*torch.tanh(self.alpha_dense) * self.ff( self.norm2(x) )  
        
        return x 





class BasicTransformerBlock(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, n_heads, d_head, fuser_type, use_checkpoint=True):
        super().__init__()
        self.attn1 = SelfAttention(query_dim=query_dim, heads=n_heads, dim_head=d_head)  
        self.ff = FeedForward(query_dim, glu=True)
        self.attn2 = CrossAttention(query_dim=query_dim, key_dim=key_dim, value_dim=value_dim, heads=n_heads, dim_head=d_head)  
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        self.norm3 = nn.LayerNorm(query_dim)
        self.use_checkpoint = use_checkpoint

        if fuser_type == "gatedSA":
            # note key_dim here actually is context_dim
            self.fuser = GatedSelfAttentionDense(query_dim, key_dim, n_heads, d_head) 
        elif fuser_type == "gatedSA2":
            # note key_dim here actually is context_dim
            self.fuser = GatedSelfAttentionDense2(query_dim, key_dim, n_heads, d_head) 
        elif fuser_type == "gatedCA":
            self.fuser = GatedCrossAttentionDense(query_dim, key_dim, value_dim, n_heads, d_head) 
        else:
            assert False 


    def forward(self, x, context, objs, tv_mask=None):
#        return checkpoint(self._forward, (x, context, objs), self.parameters(), self.use_checkpoint)
        if self.use_checkpoint and x.requires_grad:
            return checkpoint.checkpoint(self._forward, x, context, objs, tv_mask)
        else:
            return self._forward(x, context, objs, tv_mask)

    def _forward(self, x, context, objs, tv_mask=None): 
        x = self.attn1( self.norm1(x) ) + x 
        x = self.fuser(x, objs, tv_mask) # identity mapping in the beginning 
        x = self.attn2(self.norm2(x), context, context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    def __init__(self, in_channels, key_dim, value_dim, n_heads, d_head, depth=1, fuser_type=None, use_checkpoint=True):
        super().__init__()
        self.in_channels = in_channels
        query_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        
        self.proj_in = nn.Conv2d(in_channels,
                                 query_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(query_dim, key_dim, value_dim, n_heads, d_head, fuser_type, use_checkpoint=use_checkpoint)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(query_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context, objs, tv_mask=None):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, context, objs, tv_mask)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in