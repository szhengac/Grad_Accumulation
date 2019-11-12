import sys
import mxnet as mx
from mxnet.gluon import nn


class ResidualBlock(nn.HybridBlock):
    def __init__(self, model, prefix=None, params=None):
        super(ResidualBlock, self).__init__(prefix=prefix, params=params)
        self._model = model

    def hybrid_forward(self, F, x):
        return x + self._model(x)


class DotProduct(nn.HybridBlock):
    def __init__(self, units, prefix=None, params=None):
        super(DotProduct, self).__init__(prefix=prefix, params=params)
        self._query_proj = nn.Dense(units, flatten=False, use_bias=False)
        self._key_proj = nn.Dense(units, flatten=False, use_bias=False)
        self._value_proj = nn.Dense(units, flatten=False, use_bias=False)

    def hybrid_forward(self, F, x):
        query = self._query_proj(x)
        key = self._key_proj(x)
        value = self._value_proj(x)
        query = F.contrib.div_sqrt_dim(query)
        scores = F.batch_dot(query, key, transpose_b=True)
        normalized_scores = F.softmax(scores, axis=-1)
        output = F.batch_dot(normalized_scores, value)
        return output


grad_acc = bool(int(sys.argv[1]))
stop_step = int(sys.argv[2])
is_training = bool(int(sys.argv[3]))

ctx = mx.gpu()
data = [[mx.nd.array([[1, 2, 6, 1, 4, 6, 9, 12, 3, 2, 3, 5, 0, 1], [8, 3, 3, 3, 1, 0, 11, 9, 9, 3, 5, 2, 4, 3]]), 
         mx.nd.array([[9, 8, 3, 5, 0, 1, 2, 11, 7, 6, 5, 2, 5, 4], [0, 1, 2, 8, 4, 3, 2, 8, 0, 1, 1, 5, 3, 2]])],
        [mx.nd.array([[1, 2, 6, 1, 4, 6, 9, 12, 3, 2, 3, 5, 0, 1], [11, 12, 10, 9, 8, 3, 2, 7, 4, 10, 4, 8, 5, 3]]), 
         mx.nd.array([[5, 0, 1, 2, 6, 5, 12, 4, 12, 9, 0, 1, 4, 3], [1, 2, 6, 1, 4, 6, 9, 12, 3, 2, 3, 5, 0, 1]])],
        [mx.nd.array([[8, 3, 3, 3, 1, 0, 11, 9, 9, 3, 5, 2, 4, 3], [0, 1, 2, 8, 4, 3, 2, 8, 0, 1, 1, 5, 3, 2]]), 
         mx.nd.array([[0, 1, 2, 8, 4, 3, 2, 8, 0, 1, 1, 5, 3, 2], [9, 8, 3, 5, 0, 1, 2, 11, 7, 6, 5, 2, 5, 4]])]]
loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()
model = nn.HybridSequential()
embedding = nn.Embedding(13, 512)
model.add(embedding)
model.add(nn.LayerNorm())

submodel = nn.HybridSequential()
submodel.add(DotProduct(512))
submodel.add(nn.Dense(512, flatten=False, use_bias=False))
submodel.add(nn.Activation('relu'))
model.add(ResidualBlock(submodel))

model.add(nn.LayerNorm())
model.add(nn.Dense(13, flatten=False, use_bias=False))
model.initialize(init=mx.init.Xavier(magnitude=3.0), ctx=ctx)
if not is_training:
    model.load_parameters('save.params')
model.hybridize(static_alloc=True)
loss.hybridize(static_alloc=True)
trainer = mx.gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': 0.0001, 'beta2': 0.98})
if not is_training:
    if grad_acc:
        model.collect_params().setattr('grad_req', 'add')
    for v in model.collect_params().values():
        v.data()[:] /= 1
step = 0
for _ in range(10000):
    for sample, label in data:
        step += 1
        sample = sample.as_in_context(ctx)
        label = label.as_in_context(ctx)
        with mx.autograd.record():
            output = model(sample)
            ls = loss(output, label).mean()
        ls.backward()
        if is_training:
            trainer.step(1)
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
            if not is_training:
                mx.nd.waitall()
                sys.exit()
if is_training:
    model.save_parameters('save.params')
