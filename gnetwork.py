import numpy as np
import gnumpy as gnp
import time

def sigmoid(z):
    return 1/(1+gnp.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

class Network:
    def __init__(self, sizes):
        self.sizes = sizes;
        self.shapes = [(sizes[i], sizes[i+1]) for i in range(0, len(sizes)-1)]
        print 'Shapes',self.shapes
        self.weights = [gnp.garray(np.random.randn(shape[0],shape[1])) for shape in self.shapes]
        self.biases = [gnp.garray(np.random.randn(1, shape[1])) for shape in self.shapes]
        print 'Weights', [w.shape for w in self.weights]
        print 'Biases', [b.shape for b in self.biases]
        self.WGrads = [gnp.zeros(w.shape) for w in self.weights]
        self.biasGrads = [gnp.zeros(b.shape) for b in self.biases]
        self.momentum = 0

    def fprop(self, data):
        a = data
        for l in xrange(len(self.weights)):
            z = self.biases[l] + gnp.dot(a,self.weights[l])
            a = sigmoid(z)
        return a

    def bprop(self, data, targs):
        cost = 0
        a = data
        acts = [a]
        zs = [a]
        for l in xrange(len(self.weights)):
            z = self.biases[l] + gnp.dot(a,self.weights[l])
            a = sigmoid(z)
            zs.append(z)
            acts.append(a)
        #print acts[-1]
        delta_l = self.loss_fn_grad(acts[-1], targs) * sigmoid_prime(zs[-1])
        for l in reversed(xrange(len(self.weights))):
            self.biasGrads[l] += gnp.sum(delta_l, axis = 0)
            self.WGrads[l] += gnp.dot(acts[l].T, delta_l)
            if l > 0:
                delta_l = gnp.dot(delta_l, self.weights[l].T)*sigmoid_prime(zs[l])
        cost = self.loss_fn(acts[-1], targs)
        n_err = gnp.sum(acts[-1].argmax(axis=1) != targs.argmax(axis=1))
        return (cost, n_err)

    def loss_fn(self, o, y):
        #mean squared error
        e = o - y
        return gnp.sum(e*e)/2

    def loss_fn_grad(self, o, y):
        # grad of mean squared error
        return o - y

    def scale_dervs(self, scale):
        for i in xrange(len(self.weights)):
            self.WGrads[i] *= scale
            self.biasGrads[i] *= scale

    def choose_mini_batch(self, data, targs, mbsz, n):
        indx = np.arange(n*mbsz,(n+1)*mbsz)
        return data[indx], targs[indx]

    def sample_mini_batch(self, data, targs, mbsz):
        indx = np.arange(len(data))
        np.random.shuffle(indx)
        indx = indx[:mbsz]
        return data[indx], targs[indx]

    def update_mini_batch(self, data, targs, eta, lambda_w, mbsz):
        self.scale_dervs(self.momentum)
        cost, n_err  = self.bprop(data, targs)
        #cost_r = sum(np.linalg.norm(w.as_numpy_array())**2 for w in self.weights)
        #cost += cost_x + lambda_w*cost_r
        #print "Cost=%f" % cost,
        for i in xrange(len(self.weights)):
            self.biases[i] -= eta/mbsz*self.biasGrads[i]
            self.weights[i] -= eta/mbsz*(self.WGrads[i] + lambda_w*2*self.weights[i])
        return n_err
        
    def SGD(self, train_data, train_targs, eta=0.1, tau = 10., lambda_w=0.1, epochs=10, mbsz=5, test_data=gnp.garray(0), test_targs=gnp.garray(0)):
        num_iter = int(np.ceil(len(train_data)*1.0/mbsz))
        print 'num_iter/epoch=%d' % num_iter
        for epoch in range(epochs):
            print 'epoch=%d' % epoch
            eta_e = eta * tau/(tau+epoch)
            n_err = 0
            for i in xrange(num_iter):
                #data, targs = self.sample_mini_batch(train_data, train_targs, mbsz)
                data, targs = self.choose_mini_batch(train_data, train_targs, mbsz, i)
                n_err += self.update_mini_batch(data, targs, eta_e, lambda_w, mbsz)
            if test_data != None:
                v_err = self.classification_error(test_data, test_targs)
                print 'n_err_train = %d, n_err_validation = %d, train error = %f, validation error = %f' % (n_err, v_err, n_err*1./train_data.shape[0], v_err*1./test_data.shape[0])

    def classification_error(self, data, targs):
        outputs = self.fprop(data)
        n_err = gnp.sum(outputs.argmax(axis=1) != targs.argmax(axis=1))
        return n_err

if __name__ == "__main__":
    #train_data = np.random.randint(0, 255, (100, 784))
    #train_targs = np.random.randint(0, 9, (100, 1))
    #test_data = np.random.randint(0, 255, (10, 784))
    #test_targs = np.random.randint(0, 9, (10, 1))

    import mnist_loader as loader
    train_data, train_targs, test_data = loader.load_kaggle_mnist_data_v2()

    vsz = int(np.ceil(train_data.shape[0]*0.2))
    print 'Validation set size = %d' % vsz
    train_data = gnp.garray(train_data[vsz:,:])
    validation_data = gnp.garray(train_data[:vsz,:])
    train_targs = gnp.garray(train_targs[vsz:,:])
    validation_targs = gnp.garray(train_targs[:vsz,:])

    st = time.clock()
    net = Network([784, 200, 100, 50, 10])
    net.SGD(train_data, train_targs, test_data = validation_data, test_targs = validation_targs, \
            eta=5., tau=5., lambda_w = 0.0001, mbsz = 64, epochs=300)
    et = time.clock()
    print 'Time elapsed= %f seconds' % (et-st)

    predictions = net.fprop(test_data)
    predictions = [p.argmax() for p in predictions]

    import pandas as pd
    df = pd.DataFrame([{'label': p} for p in predictions])
    df.index += 1
    df.index.names = ['ImageId']
    df.to_csv('data/submission.csv')

