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
        a = x
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(b+np.dot(w,a))
        return a

    def bprop(self, x, y):
        a = x
        acts = []
        zs = []
        for bi, w in zip(self.biases, self.weights):
            z = bi + np.dot(w,a)
            a = sigmoid(z)
            zs.append(z)
            acts.append(a)
        #print acts[-1]
        gCa = self.loss_fn_grad(acts[-1], y)
        delta_l = gCa * sigmoid_prime(zs[-1])
        delta_b = [np.zeros(b.shape) for b in self.biases]
        delta_w = [np.zeros(w.shape) for w in self.weights]
        for l in xrange(len(delta_b)-1, 0, -1):
            delta_b[l] = delta_l
            delta_w[l] = np.dot(delta_l, acts[l-1].T)
            delta_l = np.dot(self.weights[l].T, delta_l)*sigmoid_prime(zs[l-1])
        cost = self.loss_fn(acts[-1], y)
        return (delta_b, delta_w, cost)

    def num_grad(self, x, y):
        eps = 1e-2
        bs = [b.copy() for b in self.biases]
        ws = [w.copy() for w in self.weights]
        delta_b = [np.zeros(b.shape) for b in self.biases]
        delta_w = [np.zeros(w.shape) for w in self.weights]
        for l in xrange(len(bs)):
            for i in xrange(bs[l].size):
                self.biases[l][i] = bs[l][i] + eps
                oh = self.fprop(x)
                eh = self.loss_fn(oh, y)
                self.biases[l][i] = bs[l][i] - eps
                ol = self.fprop(x)
                el = self.loss_fn(ol, y)
                delta_b[l][i] = (eh - el)/(2*eps)
                self.biases[l][i] = bs[l][i]
                #if i == 0 and l == len(bs)-1: print oh, ol, eh, el
            for i in xrange(ws[l].shape[0]):
                for j in xrange(ws[l].shape[1]):
                    self.weights[l][i][j] = ws[l][i][j]+eps
                    oh = self.fprop(x)
                    eh = self.loss_fn(oh, y)
                    self.weights[l][i][j] = ws[l][i][j]-eps
                    ol = self.fprop(x)
                    el = self.loss_fn(ol, y)
                    delta_w[l][i][j] = (eh - el)/(2*eps)
                    self.weights[l][i][j] = ws[l][i][j]
        return (delta_b, delta_w)

    def grad_check(self, x, y):
        ngb, ngw = self.num_grad(x, y)
        dgb, dgw, cost = self.bprop(x, y)
        eb = 0.
        ew = 0.
        for bd, bn, wd, wn in zip(dgb, ngb, dgw, ngw):
            #eb = eb + (np.linalg.norm(bd - bn))**2 / bd.size
            #ew = ew + (np.linalg.norm(wd - wn))**2 / wd.size
            eb = np.max([eb, np.absolute(bd-bn).max()])
            ew = np.max([ew, np.absolute(wd-wn).max()])
        #print 'Grad diff: Bias=%f Weights=%f' % (np.sqrt(eb), np.sqrt(ew))
        print 'Grad diff: Bias=%f Weights=%f' % (eb, ew)
        return (dgb, dgw, cost)
                
    def loss_fn(self, o, y):
        #mean squared error
        e = o - y
        return np.sum(e*e)/2
        # log likelihood
        #e = -np.log(o[y.argmax()])
        return e

    def loss_fn_grad(self, o, y):
        # grad of mean squared error
        return o - y
        #g = np.zeros(o.shape)
        #g[y.argmax()] = 1.
        #g = o - g
        #return g

    def sample_mini_batch(self, data, targs, mbsz):
        indx = np.array(range(len(data)))
        np.random.shuffle(indx)
        indx = indx[:mbsz]
        return data[indx], targs[indx]

    def update_mini_batch(self, data, targs, eta, lambda_w):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        cost = 0
        for x, y in zip(data, targs):
            delta_b, delta_w, cost_x = self.bprop(x, y)
            #delta_b, delta_w, cost_x = self.grad_check(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_b)]
            nabla_w = [nw + dnw for nw, dnw, w in zip(nabla_w, delta_w, self.weights)]
        cost_r = sum(np.linalg.norm(w)**2 for w in self.weights)
        cost += cost_x + lambda_w*cost_r
        #print "Cost=%f" % cost,
        self.biases = [b - eta*nb for b, nb in zip(self.biases, nabla_b)]
        self.weights = [w - eta*(nw + lambda_w*2*w) for w, nw in zip(self.weights, nabla_w)]
        
    def SGD(self, train_data, train_targs, eta=0.1, lambda_w=0.1, epochs=10, mbsz=5, test_data=None, test_targs=None):
        num_iter = int(np.ceil(len(train_data)*1.0/mbsz))
        print 'num_iter/epoch=%d' % num_iter
        for epoch in range(epochs):
            print 'epoch=%d' % epoch
            for i in range(num_iter):
                data, targs = self.sample_mini_batch(train_data, train_targs, mbsz)
                self.update_mini_batch(data, targs, eta, lambda_w)
            if test_data != None:
                print 'train error = %f, validation error = %f' % (self.classification_error(train_data, train_targs), 
self.classification_error(test_data, test_targs))

    def classification_error(self, test_data, test_targs):
        n_err = 0
        for x, y in zip(test_data, test_targs):
            o = self.fprop(x)
            n_err += (o.argmax() != y.argmax())
        print 'n_err=%d' % n_err,
        return n_err*1./test_data.shape[0]

    def get_predictions(self, test_data):
        os = []
        for x in test_data:
            o = self.fprop(x.reshape(x.shape[0],1))
            os.append(o)
        return os

if __name__ == "__main__":
    #train_data = np.random.randint(0, 255, (100, 784))
    #train_targs = np.random.randint(0, 9, (100, 1))
    #test_data = np.random.randint(0, 255, (10, 784))
    #test_targs = np.random.randint(0, 9, (10, 1))

    import mnist_loader as loader
    train_data, train_targs, test_data = loader.load_kaggle_mnist_data()

    vsz = int(np.ceil(train_data.shape[0]*0.2))
    print 'Validation set size = %d' % vsz
    train_data = train_data[vsz:,:]
    validation_data = train_data[:vsz,:]
    train_targs = train_targs[vsz:,:]
    validation_targs = train_targs[:vsz,:]

    net = Network([784, 100, 50, 10])
    net.SGD(train_data, train_targs, test_data = validation_data, test_targs = validation_targs, \
            eta=1./64., lambda_w = .001, mbsz = 64, epochs=30)

    predictions = net.get_predictions(test_data)
    predictions = [p.argmax() for p in predictions]

    import pandas as pd
    df = pd.DataFrame([{'label': p} for p in predictions])
    df.index.names = ['ImageId']
    df.to_csv('data/submission.csv')

