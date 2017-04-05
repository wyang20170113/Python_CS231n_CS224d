from numpy import *
import itertools
import time
import sys

# Import NN utils
from nn.base import NNBase
from nn.math import softmax, sigmoid,make_onehot
from nn.math import MultinomialSampler, multinomial_sample
from misc import random_weight_matrix


class RNNLM(NNBase):
    """
    Implements an RNN language model of the form:
    h(t) = sigmoid(H * h(t-1) + L[x(t)])
    y(t) = softmax(U * h(t))
    where y(t) predicts the next word in the sequence

    U = |V| * dim(h) as output vectors
    L = |V| * dim(h) as input vectors

    You should initialize each U[i,j] and L[i,j]
    as Gaussian noise with mean 0 and variance 0.1

    Arguments:
        L0 : initial input word vectors
        U0 : initial output word vectors
        alpha : default learning rate
        bptt : number of backprop timesteps
    """

    def __init__(self, L0, U0=None,
                 alpha=0.005, rseed=10, bptt=1):

        self.hdim = L0.shape[1] # word vector dimensions
        self.vdim = L0.shape[0] # vocab size
        param_dims = dict(H1 = (self.hdim, self.hdim),
                          H2 = (self.hdim, self.hdim),
                          W1 = (self.hdim, self.hdim),
                          U = (self.vdim,self.hdim),
                          L = (L0.shape))
                          #U = L0.shape)
        # note that only L gets sparse updates
        param_dims_sparse = dict()
        NNBase.__init__(self, param_dims, param_dims_sparse)

        #### YOUR CODE HERE ####
        self.alpha = alpha
        self.rseed = rseed
        self.bptt = bptt
        # Initialize word vectors
        # either copy the passed L0 and U0 (and initialize in your notebook)
        # or initialize with gaussian noise here
        self.params.L = L0.copy()
        self.params.H1 = eye(self.hdim)
        self.params.H2 = eye(self.hdim)
        self.params.W1 = random_weight_matrix(*self.params.W1.shape)
        if U0 == None:
            random.seed(self.rseed)
            self.params.U = random_weight_matrix(*self.params.U.shape)
        else:
            self.params.U = U0
        # Initialize H matrix, as with W and U in part 1

        #### END YOUR CODE ####


    def _acc_grads(self, xs, ys):
        """
        Accumulate gradients, given a pair of training sequences:
        xs = [<indices>] # input words
        ys = [<indices>] # output words (to predict)

        Your code should update self.grads and self.sgrads,
        in order for gradient_check and training to work.

        So, for example:
        self.grads.H += (your gradient dJ/dH)
        self.sgrads.L[i] = (gradient dJ/dL[i]) # update row

        Per the handout, you should:
            - make predictions by running forward in time
                through the entire input sequence
            - for *each* output word in ys, compute the
                gradients with respect to the cross-entropy
                loss for that output word
            - run backpropagation-through-time for self.bptt
                timesteps, storing grads in self.grads (for H, U)
                and self.sgrads (for L)

        You'll want to store your predictions \hat{y}(t)
        and the hidden layer values h(t) as you run forward,
        so that you can access them during backpropagation.

        At time 0, you should initialize the hidden layer to
        be a vector of zeros.
        """

        # Expect xs as list of indices
        #ns = self.bptt
        ns = len(xs)
        # make matrix here of corresponding h(t)
        # hs[-1] = initial hidden state (zeros)
        hs1 = zeros((ns+1, self.hdim))
        hs2 = zeros((ns+1, self.hdim))
        # predicted probas
        #ps = zeros((ns, self.vdim))
        ps = zeros((ns, self.vdim))

        #### YOUR CODE HERE ###
        ### The following code is for RNNLM, instead of EEG
        #for t in xrange(ns):
        #    hs[t] = sigmoid(dot(self.params.H, hs[t - 1]) + self.sparams.L[xs[t]])
        #    ps[t] = softmax(dot(self.params.U, hs[t]))

        ##
        # Backward propagation through time

        #for j in xrange(ns):
        #    y = make_onehot(ys[j], self.vdim)
        #    y_hat_minus_y = ps[j] - y
        #    self.grads.U += outer(y_hat_minus_y, hs[j])
        #    delta = dot(self.params.U.T, y_hat_minus_y) * hs[j] * (1.0 - hs[j])

            # start at j and go back self.bptt times (total self.bptt + 1 elements, including current one)
        #    for t in xrange(j, j - self.bptt - 1, -1):
        #        if t - 1 >= -1:
        #            self.grads.H += outer(delta, hs[t - 1])
        #            self.sgrads.L[xs[t]] = delta
        #            delta = dot(self.params.H.T, delta) * hs[t - 1] * (1.0 - hs[t - 1])
        ################################################################################
        
        for t in xrange(ns):
            hs1[t] = tanh(dot(self.params.H1, hs1[t - 1]) + self.params.L[xs[t],:])
            hs2[t] = tanh(self.params.H2.dot(hs2[t-1]) + self.params.W1.dot(hs1[t]))
            ps[t] = softmax(dot(self.params.U, hs2[t]))
            
        #ps = softmax(self.params.U.dot(hs[self.bptt - 1]))

        ##
        # Backward propagation through time
        #yt = repeat(ys,ns)
        yt = ys
        for j in xrange(ns):
            y = make_onehot(yt[j], self.vdim)
            y_hat_minus_y = ps[j] - y
            self.grads.U += outer(y_hat_minus_y, hs2[j])
            delta2 = dot(self.params.U.T, y_hat_minus_y) * ( 1.0 - hs2[j]**2)
            delta1 = (self.params.W1.T.dot(delta2)) * (1.0 - hs1[j]**2)
            # start at j and go back self.bptt times (total self.bptt + 1 elements, including current one)
            for t in xrange(j,j - self.bptt - 1, -1):
                if t - 1 >= -1:
                    self.grads.H2 += outer(delta2, hs2[t - 1])
                    self.grads.W1 += outer(delta2,hs1[t])
                    #delta2 = dot(self.params.H2.T, delta2) * (1.0 - hs2[t - 1]**2)
                    #delta1 = (self.params.W1.T.dot(delta2)) * (1.0 - hs1[t]**2)
                    #self.grads.L[yt[t]] += outer(delta1,xs)
                    self.grads.L[xs[t]] += delta1
                    self.grads.H1 += outer(delta1,hs1[t-1])
                    delta2 = dot(self.params.H2.T, delta2) * (1.0 - hs2[t - 1]**2)
                    delta1 = (dot(self.params.H1.T, delta1) + self.params.W1.T.dot(delta2)) * (1.0 - hs1[t - 1]**2)

        #### END YOUR CODE ####



    def grad_check(self, x, y, outfd=sys.stderr, **kwargs):
        """
        Wrapper for gradient check on RNNs;
        ensures that backprop-through-time is run to completion,
        computing the full gradient for the loss as summed over
        the input sequence and predictions.

        Do not modify this function!
        """
        bptt_old = self.bptt
        self.bptt = len(y)
        print >> outfd, "NOTE: temporarily setting self.bptt = len(y) = %d to compute true gradient." % self.bptt
        NNBase.grad_check(self, x, y, outfd=outfd, **kwargs)
        self.bptt = bptt_old
        print >> outfd, "Reset self.bptt = %d" % self.bptt

    #def predict(self,xs,ys,pred):
    #    idx = 0
    #    for x,y in itertools.izip(xs,ys):
    #        ns = len(x)
    #        hs = zeros((ns+1, self.hdim))
            # predicted probas
    #        ps = zeros((ns, self.vdim))
        
    #        for t in xrange(ns):
    #            hs[t] = sigmoid(dot(self.params.H, hs[t - 1]) + self.sparams.L[y] * x[t])
    #            ps[t] = softmax(dot(self.params.U, hs[t]))
    #        pred[idx] = argmax(ps[ns-1])
    #        idx += 1
    #    return pred
    
    def predict(self,xs,pred):
        #idx = 0
        hs = zeros((self.bptt+1, self.hdim))
        # predicted probas
        ps = zeros((self.vdim,))
        #ps = zeros((1, self.vdim))

        #### YOUR CODE HERE ###
        
        for i in xrange(xs.shape[0]):
            for t in xrange(self.bptt):
                hs[t] = tanh(dot(self.params.H, hs[t - 1]) + self.params.L.dot(xs[i]))
            #ps[t] = softmax(dot(self.params.U, hs[t]))
            #ps[t] = softmax(self.params.U.dot(hs[t]))
            ps = softmax(self.params.U.dot(hs[self.bptt-1]))
            pred[i] = argmax(ps)
            #idx += 1
        return pred
            
        
    def compute_seq_loss(self, xs, ys):
        """
        Compute the total cross-entropy loss
        for an input sequence xs and output
        sequence (labels) ys.

        You should run the RNN forward,
        compute cross-entropy loss at each timestep,
        and return the sum of the point losses.
        """

        J = 0
        #ns = self.bptt
        ns = len(xs)
        # make matrix here of corresponding h(t)
        # hs[-1] = initial hidden state (zeros)
        hs1 = zeros((ns+1, self.hdim))
        hs2 = zeros((ns+1, self.hdim))
        # predicted probas
        ps = zeros((ns, self.vdim))
        #yt = repeat(ys,ns)
        yt = ys
        for t in xrange(ns):
            hs1[t] = tanh(dot(self.params.H1, hs1[t - 1]) + self.params.L[xs[t],:])
            hs2[t] = tanh(self.params.H2.dot(hs2[t-1]) + self.params.W1.dot(hs1[t]))
            ps[t] = softmax(dot(self.params.U, hs2[t]))
            J += -log(ps[t][yt[t]])
        #ps = softmax(self.params.U.dot(hs[self.bptt - 1]))

        ##
        # Backward propagation through time
        
        #### YOUR CODE HERE ###
        #for t in xrange(self.bptt):
        #    hs[t] = tanh(dot(self.params.H, hs[t - 1]) + self.params.L.dot(xs))
            #ps[t] = softmax(dot(self.params.U, hs[t]))
        #    ps[t] = softmax(self.params.U.dot(hs[t]))
        #    J += -log(ps[t][yt[t]])
        #### END YOUR CODE ####
        return J


    def compute_loss(self, X, Y):
        """
        Compute total loss over a dataset.
        (wrapper for compute_seq_loss)

        Do not modify this function!
        """
        if not isinstance(X[0], ndarray): # single example
            return self.compute_seq_loss(X, Y)
        else: # multiple examples
            return sum([self.compute_seq_loss(xs,ys)
                       for xs,ys in itertools.izip(X, Y)])

    def compute_mean_loss(self, X, Y):
        """
        Normalize loss by total number of points.

        Do not modify this function!
        """
        J = self.compute_loss(X, Y)
        ntot = sum(map(len,Y))
        #ntot = Y.shape[0]
        return J / float(ntot)
        #return J


    def generate_sequence(self, init, end, maxlen=100):
        """
        Generate a sequence from the language model,
        by running the RNN forward and selecting,
        at each timestep, a random word from the
        a word from the emitted probability distribution.

        The MultinomialSampler class (in nn.math) may be helpful
        here for sampling a word. Use as:

            y = multinomial_sample(p)

        to sample an index y from the vector of probabilities p.


        Arguments:
            init = index of start word (word_to_num['<s>'])
            end = index of end word (word_to_num['</s>'])
            maxlen = maximum length to generate

        Returns:
            ys = sequence of indices
            J = total cross-entropy loss of generated sequence
        """

        J = 0 # total loss
        ys = [init] # emitted sequence

        #### YOUR CODE HERE ####


        #### YOUR CODE HERE ####
        return ys, J



class ExtraCreditRNNLM(RNNLM):
    """
    Implements an improved RNN language model,
    for better speed and/or performance.

    We're not going to place any constraints on you
    for this part, but we do recommend that you still
    use the starter code (NNBase) framework that
    you've been using for the NER and RNNLM models.
    """

    def __init__(self, *args, **kwargs):
        #### YOUR CODE HERE ####
        raise NotImplementedError("__init__() not yet implemented.")
        #### END YOUR CODE ####

    def _acc_grads(self, xs, ys):
        #### YOUR CODE HERE ####
        raise NotImplementedError("_acc_grads() not yet implemented.")
        #### END YOUR CODE ####

    def compute_seq_loss(self, xs, ys):
        #### YOUR CODE HERE ####
        raise NotImplementedError("compute_seq_loss() not yet implemented.")
        #### END YOUR CODE ####

    def generate_sequence(self, init, end, maxlen=100):
        #### YOUR CODE HERE ####
        raise NotImplementedError("generate_sequence() not yet implemented.")
        #### END YOUR CODE ####