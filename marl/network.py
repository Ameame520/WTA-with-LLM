"""M2 + M3: set-attention encoder and decision head (PyTorch).

M2 SetEncoder:
    - input projections x_j(10->64), q_i(5->64), g(3->64);
    - ONE multi-head self-attention layer (2 heads, d=64) over the union
      of target nodes + platform query + global node (target-set part is
      permutation invariant by construction);
    - outputs platform embedding z_i, target keys/values (k_j, v_j),
      global embedding z_g.

M3 DecisionHead:
    - engagement score  s_j = MLP2([z_i; k_j; z_i*k_j])  (192->32->1)
    - hold score        s_p = MLP2([z_i; z_g])           (128->32->1)
    - distribution = softmax([s_p, s_j]/tau) over {hold} U J_feas

Parameter budget: ~2.6e4 trainable params (hard red line < 1e5, asserted
by assert_params()).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

D_MODEL = 64
N_HEADS = 2
X_DIM, Q_DIM, G_DIM = 10, 5, 3


class MarlNet(nn.Module):
    def __init__(self, d: int = D_MODEL, n_heads: int = N_HEADS):
        super().__init__()
        self.d = d
        self.x_proj = nn.Linear(X_DIM, d)
        self.q_proj = nn.Linear(Q_DIM, d)
        self.g_proj = nn.Linear(G_DIM, d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
        # M3 decision head
        self.score = nn.Sequential(nn.Linear(3 * d, 32), nn.ReLU(),
                                   nn.Linear(32, 1))
        self.hold = nn.Sequential(nn.Linear(2 * d, 32), nn.ReLU(),
                                  nn.Linear(32, 1))

    # ------------------------------------------------------------------
    def encode(self, x, q, g):
        """x: [L,10] target features, q: [5] platform, g: [3] global.

        Returns (z_i [d], k [L,d], v [L,d], z_g [d]).
        """
        L = x.shape[0]
        xe = self.x_proj(x).unsqueeze(0)              # [1,L,d]
        qe = self.q_proj(q).view(1, 1, -1)            # [1,1,d]
        ge = self.g_proj(g).view(1, 1, -1)            # [1,1,d]
        tokens = torch.cat([xe, qe, ge], dim=1)       # [1,L+2,d]
        h, _ = self.attn(tokens, tokens, tokens,
                         need_weights=False)
        h = self.norm(tokens + h)
        k, v = h[0, :L], h[0, :L]                    # target k/v
        z_i = h[0, L]
        z_g = h[0, L + 1]
        return z_i, k, v, z_g

    # ------------------------------------------------------------------
    def forward(self, x, q, g, tau: float = 1.0):
        """Full network for ONE agent: returns logits [L+1] with the hold
        position FIRST (index 0), pre-softmax (temperature NOT applied -
        the caller applies tau or samples; kept raw for log-prob exactness).
        """
        z_i, k, v, z_g = self.encode(x, q, g)
        L = k.shape[0]
        if L == 0:
            s_hold = self.hold(torch.cat([z_i, z_g]))[0]
            return torch.stack([s_hold]), z_i, k, v, z_g
        zi_rep = z_i.unsqueeze(0).expand(L, -1)                    # [L,d]
        s_j = self.score(torch.cat([zi_rep, k, zi_rep * k],
                                   dim=-1)).squeeze(-1)            # [L]
        s_p = self.hold(torch.cat([z_i, z_g]))[0]                  # scalar
        return torch.cat([s_p.unsqueeze(0), s_j]), z_i, k, v, z_g

    # ------------------------------------------------------------------
    def params_count(self) -> int:
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad)


def assert_params(model: MarlNet, lo: float = 1e3, hi: float = 1e5):
    n = model.params_count()
    assert lo <= n <= hi, \
        "trainable params %d outside red-line [%g, %g]" % (n, lo, hi)
    return n


# ----------------------------------------------------------------------
# selftest: permutation consistency + parameter budget
# ----------------------------------------------------------------------

def _selftest():
    torch.manual_seed(0)
    net = MarlNet()
    n = assert_params(net)
    L = 7
    x = torch.randn(L, X_DIM)
    q = torch.randn(Q_DIM)
    g = torch.randn(G_DIM)
    logits1, _, _, _, _ = net(x, q, g)

    # random permutation of the target set: per-target logits must map
    perm = torch.randperm(L)
    logits2, _, _, _, _ = net(x[perm], q, g)
    assert torch.allclose(logits1[1:], logits2[1:][perm.argsort()],
                          atol=1e-5), "permutation inconsistency"
    assert torch.allclose(logits1[0], logits2[0], atol=1e-6)

    # empty target set: only hold logit
    logits0, *_ = net(torch.zeros(0, X_DIM), q, g)
    assert logits0.shape == (1,)

    # batched forward over 3 platforms (stacked loop - keep it simple)
    _ = [net(x, torch.randn(Q_DIM), g) for _ in range(3)]

    print("marl/network.py selftest: ALL PASS (params=%d)" % n)


if __name__ == "__main__":
    _selftest()
