# ReSPlus

ReSPlus is SPlus but uses reciprocal instead of sign, and tracks cautious momentum in projected space.

Here is what it does.

1. projects gradients to shampoo's eigenbasis;
2. updates EMA of projected gradients;
3. takes that EMA, clips magnitudes to (0.01, 10), and takes their reciprocals;
4. unprojects resulting update;
5. tracks EMA of resulting updates, the update is grafted to that EMA for stability. This prevents the update from changing norm quickly.
6. tracks EMA of model weights. This always improves test loss in my experiments, though ReSPlus also outperforms SPlus when weight EMA is disabled in both.

## Usage example

```py
from torchalgos.experimental import ReSPlus

opt = ReSPlus(
    # you can pass `model.parameters()`
    # but pass `model` to automatically disable preconditioning for embeddings
    model,
    lr=3e-3,
    shampoo_beta=0.95 # try 0.95 and 0.0
)

model.train()
opt.train() # this is important for ReSPlus, it switches between train params and weight EMAs

for inputs, targets in dl_train:
    preds = model(inputs)
    loss = loss_fn(preds, targets)
    opt.zero_grad()
    loss.backward()
    opt.step()

model.eval()
opt.eval()
# do test epoch here
```

## Benchmarks

I tune learning rates for all optimizers for a fair comparison, and also run for a relatively large number of steps, so running benchmarks takes a while, so to keep times realistic I benchmark on 3 tiny models - MLP, RNN and ConvNet. It beats all other opts on MLP and RNN with `shampoo_beta=0.95`, and RNN and ConvNet with `shampoo_beta=0`. Though with `shampoo_beta=0` it's clearly unstable as seen by the jagged learning rate search curve.

BENCH 1
BENCH 2
BENCH 3

The curves look weird, that's because this is a weird optimizer. Taking the reciprocal makes it go along the walls of the valley, rather than along the floor. Look at the way it minimizes nesterov's piecewise rosenbrock (which is btw a hard function and many opts fail to minimize it):

PIC

**Note: it might not scale to bigger models. Its a weird optimizer and it's unpredictable what would happen with other models. I might test when I'm not too lazy.**

## Reasoning behind the update rule

This is a very old idea I had that you should make small careful steps when gradient is big or you end up somewhere completely different, and big steps when gradient is small because otherwise you will make too little progress. It's a pretty stupid idea and doesn't really work, but I tried it inside Shampoo's eigenbasis and it ended up behaving in a way that somehow actually works well.

I suspect that it works well because it goes all over the place which somehow acts as a regularizer.

# Other optimizers

This also implements other optimizers implemented here, SOAP and SPlus are very good implementations and thoroughly tested by way of me always using them when I train models. Feel free to use them.

There are also a whole bunch of my other experiments and most of them suck so you probably shouldn't use them.
