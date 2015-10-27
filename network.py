import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

class Network:
    def __init__(self, sizes):
        self.sizes = sizes;
        self.shapes = [(sizes[i], sizes[i+1]) for i in range(0, len(sizes)-1)]
        print 'Shapes',self.shapes
        self.weights = [np.random.randn(shape[1],shape[0]) for shape in self.shapes]
        self.biases = [np.random.randn(shape[1], 1) for shape in self.shapes]
        print 'Weights', [w.shape for w in self.weights]
        print 'Biases', [b.shape for b in self.biases]

    def fprop(self, x):
        a = np.copy(x)
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(b+np.dot(w,a))
        return a

    def bprop(self, x, y):
        a = x.reshape((x.shape[0],1))
        acts = [a]
        zs = [np.zeros(a.shape)]
        for bi, w in zip(self.biases, self.weights):
            z = bi + np.dot(w,a)
            a = sigmoid(z)
            zs.append(z)
            acts.append(a)
        gCa = self.loss_fn(acts[-1], y)
        delta_l = gCa * sigmoid_prime(zs[-1])
        delta_b = [np.zeros(b.shape) for b in self.biases]
        delta_w = [np.zeros(w.shape) for w in self.weights]
        for l in xrange(len(delta_b)-1, 0, -1):
            delta_b[l] = delta_l
            delta_w[l] = np.dot(delta_l, acts[l].T)
            delta_l = np.dot(w.T, delta_l)*sigmoid_prime(zs[l])
        return (delta_b, delta_w)

    def loss_fn(self, o, y):
        return o - y

    def sample_mini_batch(self, data, targs, mbsz):
        indx = np.array(range(len(data)))
        np.random.shuffle(indx)
        indx = indx[:mbsz]
        return data[indx], targs[indx]

    def update_mini_batch(self, data, targs, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in zip(data, targs):
            delta_b, delta_w = self.bprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_w)]
        self.biases = [b - eta*nb for b, nb in zip(self.biases, nabla_b)]
        self.weights = [w - eta*nw for w, nw in zip(self.weights, nabla_w)]
        
    def SGD(self, train_data, train_targs, eta=0.1, epochs=10, mbsz=5, test_data=None, test_targs=None):
        num_iter = int(np.ceil(len(train_data)*1.0/mbsz))
        print 'num_iter/epoch=%d' % num_iter
        for epoch in range(epochs):
            print 'epoch=%d' % epoch
            for i in range(num_iter):
                data, targs = self.sample_mini_batch(train_data, train_targs, mbsz)
                self.update_mini_batch(data, targs, eta)

if __name__ == "__main__":
    print sigmoid(np.array(range(-5,5,1)))
    print sigmoid_prime(np.array(range(-5,5,1)))
    net = Network([784, 60, 10])
    print net.fprop(np.random.randn(784,1))

    train_data = np.random.randint(0, 255, (100, 784))
    train_targs = np.random.randint(0, 9, (100, 1))
    test_data = np.random.randint(0, 255, (10, 784))
    test_targs = np.random.randint(0, 9, (10, 1))

    net.SGD(train_data, train_targs)

