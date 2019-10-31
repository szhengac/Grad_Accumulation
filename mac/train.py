import sys
import mxnet as mx
from mxnet.gluon import nn

grad_acc = bool(int(sys.argv[1]))
stop_step = int(sys.argv[2])
ctx = mx.cpu()

data = [[mx.nd.array([0.132131, 0.0054324134, -1.05325, 2.042315, 3.342350, -0.552351, 0.00151235, 1.005235, 2.053253, -0.00004234]),
         mx.nd.array((0,))],
        [mx.nd.array([0.132131, 0.0054324134, -1.05325, 2.042315, 3.342350, -0.552351, 0.00151235, 1.005235, 2.053253, -0.00004234]),
        mx.nd.array((1,))]]
loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()
model = nn.HybridSequential()
model.add(nn.Dense(1000))
model.add(nn.Activation('relu'))
model.add(nn.Dense(2))
model.initialize(init=mx.init.Xavier(magnitude=3.0), ctx=ctx)
model.load_parameters('save.params')
model.hybridize(static_alloc=True)
loss.hybridize(static_alloc=True)
trainer = mx.gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': 0.1})
if grad_acc:
    model.collect_params().setattr('grad_req', 'add')
for v in model.collect_params().values():
    v.data()[:] /= 1e6
step = 0
for _ in range(1000):
    for sample, label in data:
        step += 1
        sample = mx.nd.expand_dims(sample, axis=0).as_in_context(ctx)
        label = mx.nd.expand_dims(label, axis=0).as_in_context(ctx)
        with mx.autograd.record():
            output = model(sample)
            ls = loss(output, label).sum()
        ls.backward()
        if not grad_acc:
            if step == 1:
                grads = {k: p.grad(ctx).copy() for k, p in model.collect_params().items()}
            else:
                for k, v in model.collect_params().items():
                    grads[k][:] += v.grad(ctx)
        if step == stop_step:
            if grad_acc:
                grads = {k: p.grad(ctx) for k, p in model.collect_params().items()}
                mx.nd.save('grad_acc.params', grads)
            else:
                mx.nd.save('grad.params', grads)
            mx.nd.waitall()
            sys.exit()