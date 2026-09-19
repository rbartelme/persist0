import torch
from torch.autograd import gradcheck

from persist0 import TopoH0Loss, h0_persistence


def _sym(x):
    D = torch.cdist(x, x)
    return (D + D.transpose(1, 2)) / 2


def test_deaths_gradcheck():
    torch.manual_seed(0)
    x = torch.randn(2, 6, 3, dtype=torch.float64, requires_grad=True)
    fn = lambda x: h0_persistence(_sym(x))[1]
    assert gradcheck(fn, (x,), eps=1e-6, atol=1e-4)


def test_loss_decreases():
    torch.manual_seed(0)
    x_ref = torch.randn(1, 32, 4)
    x = torch.randn(1, 32, 4, requires_grad=True)
    loss_fn = TopoH0Loss()
    opt = torch.optim.Adam([x], lr=1e-2)
    first = None
    for _ in range(200):
        opt.zero_grad()
        loss = loss_fn(_sym(x), _sym(x_ref))
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < 0.5 * first
