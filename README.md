<h1 align='center'>Torchalgos</h1>

This implements SOAP, SPlus with modifications to improve them, and a bunch of experimental optimizers.

```py
from torchalgos import SOAP

opt = SOAP(
    # you can pass `model.parameters()`
    # but pass `model` to automatically disable preconditioning for embeddings
    model,
    lr=3e-3,
    ema_rate=0.999, # use weight averaging (big free boost to val score)
)

model.train()
opt.train() # use this if you enabled weight averaging, it switches between train params and the EMA

for inputs, targets in dl_train:
    preds = model(inputs)
    loss = loss_fn(preds, targets)
    opt.zero_grad()
    loss.backward()
    opt.step()

model.eval()
opt.eval()
# do test epoch here after calling opt.eval() to switch to averaged weights
```

## Implemented optimizers

### SOAP

<https://arxiv.org/abs/2409.11321>

Runs Adam in Shampoo's eigenbasis. We use POGO (<https://github.com/adrianjav/pogo>) by default to track the eigenbasis without any decompositions. But you can switch to subspace iteration or eigendecomposition.

We also have grafting to update EMA enabled by default which usually makes SOAP more stable and faster to train.

1d params are preconditiond by default. For big model you can disable this to use much less memory. You can also choose which dims to precondition, for example:

```py
opt = SOAP(model, lr=3e-3, precondition_1d=False, precond_dims=-2)
```

-2 is second to last dim which is the output dim on linear and conv weights, which works well and uses less memory.

### SPlus

https://arxiv.org/abs/2506.07254

Projects gradient EMA to Shampoo's eigenbasis, takes it's sign and unprojects. Canonical SPlus only applies to 2D params (linears) whereas we apply it to all tensors by default like Shampoo. Like SOAP, you can choose `precond_dims` and `precondition_1d` to save some memory. SPlus has `ema_rate=0.999` by default as per the paper.

### RePlus

This is an optimizer I have devised that beats SOAP and SPlus in many benchmarks. It projects gradients to Shampoo's eigenbasis, computes a clamped reciprocal of their cautious EMA and unprojects.

```
from torchalgos.experimental import RePlus

opt = RePlus(model, lr=3e-3, shampoo_beta=0.95) # try shampoo_beta=0.95 and 0.0
# use opt.train() and opt.eval() with model.train() and model.eval()
```
For more info and benchmarks: https://github.com/inikishev/torchalgos/blob/main/RePlus.md

### Other optimizers

There are also a whole bunch of my other experiments, you probably shouldn't use them because they aren't that good.
