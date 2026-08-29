# RePlus

RePlus is SPlus but uses reciprocal instead of sign, and tracks cautious momentum in projected space.

Here is what it does.

1. projects gradients to shampoo's eigenbasis;
2. updates EMA of projected gradients;
3. takes that EMA, clips magnitudes to (0.01, 10), and takes their reciprocals, and applies cautioning;
4. unprojects resulting update;
5. tracks EMA of resulting updates, the update is grafted to that EMA for stability. This prevents the update from changing norm quickly.
6. tracks EMA of model weights. This always improves test loss in my experiments, though RePlus also outperforms SPlus when weight EMA is disabled in both.

## Usage example

```py
from torchalgos.experimental import RePlus

opt = RePlus(
    # you can pass `model.parameters()`
    # but pass `model` to automatically disable preconditioning for embeddings
    model,
    lr=3e-3,
    shampoo_beta=0.95 # try 0.95 and 0.0
)

model.train()
opt.train() # this is important for RePlus, it switches between train params and weight EMAs

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

I tune learning rates for all optimizers for a fair comparison, and also run for a relatively large number of steps, so running benchmarks takes a while, so to keep times realistic I benchmark on mini-batch training of 3 tiny mnist1d models - MLP, RNN and ConvNet. It beats all other opts on MLP and RNN with `shampoo_beta=0.95`, and RNN and ConvNet with `shampoo_beta=0`. Though with `shampoo_beta=0` it's clearly unstable as seen by the jagged learning rate search curve.

<img width="3520" height="835" alt="image" src="https://github.com/user-attachments/assets/36d1351b-6b6b-4790-900c-06f2b9f81586" />

<img width="3486" height="839" alt="image" src="https://github.com/user-attachments/assets/1de47c15-101c-4ab6-bc1a-3fbe749a2437" />

<img width="3486" height="832" alt="image" src="https://github.com/user-attachments/assets/ca8c2870-a397-4d0a-92a7-59bc425cf5d6" />


**Note: no idea if it works well on bigger models. I might test when I'm not too lazy. But I also tried TrOCR-base and I don't have enough VRAM for even one sided SPlus so I can't really test it.**