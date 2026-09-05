<h1 align='center'>Torchalgos</h1>

This implements SOAP, SPlus with multiple improvements, and a bunch of experimental optimizers.

They are used as any other pytorch optimizer but if you use weight averaging, call `optimizer.train()` and `optimizer.eval()` alongside `model.train()` and `model.eval()`.

```py
from torchalgos import SOAP

opt = SOAP(
    # you can pass `model.parameters()`
    # but pass `model` to automatically use Adam for embeddings
    model,
    lr=3e-3,
    ema_rate=0.999, # use weight averaging (big free boost to val score)
)

model.train()
opt.train() # use this if set ema_rate, it switches train params and EMA

for inputs, targets in dl_train:
    preds = model(inputs)
    loss = loss_fn(preds, targets)
    opt.zero_grad()
    loss.backward()
    opt.step()

model.eval()
opt.eval() # switches to averaged weights

# ... test epoch after calling opt.eval()
```

## Implemented optimizers

### SOAP

<https://arxiv.org/abs/2409.11321>

Runs Adam in Shampoo's eigenbasis. We use POGO (<https://github.com/adrianjav/pogo>) by default to track the eigenbasis without any decompositions. But you can switch to subspace iteration or eigendecomposition.

We also have grafting to update EMA enabled by default which usually makes SOAP more stable and faster to train.

1d params are preconditiond by default. For big model you can disable this (use Adam for them) to use much less memory. You can also choose which dims to precondition, for example:

```py
opt = SOAP(model, lr=3e-3, precondition_1d=False, precond_dims=-2)
```

-2 is second to last dim which is the output dim on linear and conv weights, which works well and uses less memory.

### SPlus

<https://arxiv.org/abs/2506.07254>

Projects gradient EMA to Shampoo's eigenbasis, takes it's sign and unprojects. Canonical SPlus only applies to 2D params (linears) whereas we apply it to all tensors by default like Shampoo. Like SOAP, you can choose `precond_dims` and `precondition_1d` to save some memory. SPlus has `ema_rate=0.999` by default as per the paper.

## Experimental optimizers

### RePlus

This is an optimizer I have devised that beats SOAP and SPlus in many benchmarks. It projects gradients to Shampoo's eigenbasis, computes a clamped reciprocal of their cautious EMA and unprojects.

```py
from torchalgos.experimental import RePlus

opt = RePlus(model, lr=3e-3, shampoo_beta=0.95) # try shampoo_beta=0.95 and 0.0
# use opt.train() and opt.eval() with model.train() and model.eval()
```

For more info and benchmarks: <https://github.com/inikishev/torchalgos/blob/main/RePlus.md>

### CustomSOAP/CustomSPlus

This is SOAP/SPlus but you can replace multiplication, sum and linear interpolation in updating the covariance with any other operations like max, and can also choose the correction to the covariance matrix (it can even be asymmetric if using POGO solver).

```py
from torchalgos.experimental import CustomSOAP
add_max_soap = CustomSOAP(params, operation=torch.add, reduce=torch.max) # tropical soap
```

### DecoupleSVD

Optimize U, S and V factors of weight matrices separately, while also trying to keep U and V orthogonal. You can choose any base optimizer for those matrices, for example <https://github.com/adrianjav/pogo> on U and V to force them to be orthogonal. I think it's a promising idea but there are a lot of hyperparameters to tune and I haven't came up with any good ones.

```py
from torchalgos.experimental import CustomSOAP
from pogo import base, POGO

optimizer = DecoupleSVD(
    model.parameters(),
    opt_U = lambda params: POGO(params, base.VectorAdam(), 1e-3), # optimizer for left singular vecs
    opt_S = lambda params: torch.optim.Adam(params, 1e-4), # optimizer for singular values
    opt_Vh = lambda params: POGO(params, base.VectorAdam(), 1e-3), # optimizer for right singular vecs
    opt_dir = lambda params: torch.optim.Adam(params, 1e-3), # optimizer for direction for non-matrix params
    opt_magn = lambda params: torch.optim.Adam(params, 1e-4), # optimizer for magnitudes for non-matrix params
)
```

### DSOAP

This uses loss values at distant parameters to approximate gradients for covariance corrections, with the idea being that it will smooth it out, but it's not working well. Requires closure with `backward` argument and uses one extra forward pass per step.

```py
def closure(backward=True):
    preds = model(inputs)
    loss = loss_fn(preds, targets)
    if backward:
        opt.zero_grad()
        loss.backward()
    return loss
```

### QSOAP

This maintains a quasi-newton hessian approximation instead of covariance using one additional forward and backward pass per step, where gradients are evaluated on current batch but at previous parameters. Requires a closure. This matches SOAP in performance, so extra forward and backward passes don't seem to be worth it.