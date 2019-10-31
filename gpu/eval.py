import mxnet as mx

grad = mx.nd.load('grad.params')
grad_acc = mx.nd.load('grad_acc.params')
for k, v in grad.items():
    v2 = grad_acc[k]
    diff = v - v2
    if mx.nd.max(mx.nd.abs(diff)).asscalar() > 0:
        print(mx.nd.abs(v2))
        print(k, 'rtol:{}%, atol:{}'.format(
              mx.nd.max(mx.nd.abs(diff)/mx.nd.abs(v2)).asscalar()*100,
              mx.nd.max(mx.nd.abs(diff)).asscalar()))
