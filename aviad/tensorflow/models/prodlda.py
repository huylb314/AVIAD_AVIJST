import numpy as np
import tensorflow as tf


def xavier_init(fan_in, fan_out, constant=1):
    low = -constant*tf.sqrt(6.0/(fan_in + fan_out))
    high = constant*tf.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


def log_dir_init(fan_in, fan_out, topics=50):
    return tf.log((1.0/topics)*tf.ones([fan_in, fan_out]))

tf.reset_default_graph()
class ProdLDA(object):
    """
    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """

    def __init__(self, n_hidden_enc_1, n_hidden_enc_2, n_hidden_dec_1, n_input, n_z,
                 data_prior, act=tf.nn.softplus, ld=20.0, al=1.0, lr=0.001, bs=100, dr=0.6):
        self.n_hidden_enc_1, self.n_hidden_enc_2 = n_hidden_enc_1, n_hidden_enc_2
        self.n_hidden_dec_1 = n_hidden_dec_1
        self.n_input, self.n_z = n_input, n_z

        self.act = act
        self.learning_rate = lr
        self.ld = ld
        self.al = al
        self.bs = bs
        self.dr = dr

        # tf Graph input
        self.prior, self.prior_indices = data_prior
        self.prior_bin = tf.placeholder(tf.bool, [None, n_input, n_z])
        self.bx = tf.placeholder(tf.int32)
        self.x = tf.placeholder(tf.float32, [None, n_input])
        self.x_bin = self._get_data_bin(self.x)
        self.keep_prob = tf.placeholder(tf.float32)
        self.lambda_ = tf.placeholder(tf.float32)

        self.a = self.al*np.ones((1, self.n_z)).astype(np.float32)
        self.mu2 = tf.constant((np.log(self.a).T-np.mean(np.log(self.a), 1)).T)
        self.var2 = tf.constant((((1.0/self.a)*(1 - (2.0/self.n_z))).T +
                                 (1.0/(self.n_z*self.n_z))*np.sum(1.0/self.a, 1)).T)

        self._create_network()
        self._create_loss_optimizer()

        init = tf.initialize_all_variables()

        self.sess = tf.InteractiveSession()
        self.sess.run(init)

    def _create_network(self):
        self.network_weights = self._initialize_weights(
            self.n_hidden_enc_1, self.n_hidden_enc_2,
            self.n_hidden_dec_1, self.n_input, self.n_z
        )
        self.z_mean, self.z_log_sigma_sq = \
            self._encoder_network(self.network_weights["weights_enc"], 
                                  self.network_weights["biases_enc"])
        eps = tf.random_normal((self.bx, self.n_z), 0, 1, dtype=tf.float32)
        self.z = tf.add(self.z_mean,
                        tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))
        self.sigma = tf.exp(self.z_log_sigma_sq)

        self.x_reconstr_mean = \
            self._decoder_network(self.z, self.network_weights["weights_dec"])

    def _initialize_weights(self, n_hidden_enc_1, n_hidden_enc_2, n_hidden_dec_1, n_input, n_z):
        all_weights = dict()
        all_weights['weights_enc'] = {
            'h1': tf.get_variable('h1', [n_input, n_hidden_enc_1], initializer=tf.contrib.layers.xavier_initializer()),
            'h2': tf.get_variable('h2', [n_hidden_enc_1, n_hidden_enc_2], initializer=tf.contrib.layers.xavier_initializer()),
            'out_mean': tf.get_variable('out_mean', [n_hidden_enc_2, n_z], initializer=tf.contrib.layers.xavier_initializer()),
            'out_log_sigma': tf.get_variable('out_log_sigma', [n_hidden_enc_2, n_z], initializer=tf.contrib.layers.xavier_initializer())
        }
        all_weights['biases_enc'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_enc_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_enc_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))
        }
        all_weights['weights_dec'] = {
            'h1': tf.Variable(xavier_init(n_z, n_hidden_dec_1))
        }
        return all_weights

    def _encoder_network(self, weights, biases):
        layer_1 = self.act(tf.add(tf.matmul(self.x, weights['h1']), biases['b1']))
        layer_2 = self.act(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
        layer_enc_do = tf.nn.dropout(layer_2, self.keep_prob)

        z_mean = tf.contrib.layers.batch_norm(tf.add(tf.matmul(layer_enc_do, weights['out_mean']), biases['out_mean']))
        z_log_sigma_sq = \
            tf.contrib.layers.batch_norm(tf.add(tf.matmul(layer_enc_do, weights['out_log_sigma']), biases['out_log_sigma']))

        return (z_mean, z_log_sigma_sq)

    def _decoder_network(self, z, weights):
        layer_dec_do = tf.nn.dropout(tf.nn.softmax(z), self.keep_prob)
        x_reconstr_mean = tf.nn.softmax(tf.contrib.layers.batch_norm(tf.add(tf.matmul(layer_dec_do, weights['h1']), 0.0)))
        return x_reconstr_mean

    def _prior_loss(self, prior, weights):
        W, b = weights[0], weights[1]
        W1 = self.act(tf.add(W['h1'], b['b1']))
        W2 = self.act(tf.add(tf.matmul(W1, W['h2']), b['b2']))
        Wdr = tf.nn.dropout(W2, self.keep_prob)
        Wo = tf.nn.softmax(tf.contrib.layers.batch_norm(tf.add(tf.matmul(Wdr, W['out_mean']), b['out_mean'])))

        self.Wo = Wo
        self.prior_loss = self.lambda_ * tf.reduce_sum((prior - self.x_bin*Wo)**2, [1, 2])
        return self.prior_loss

    def _create_loss_optimizer(self):
        self.x_reconstr_mean += 1e-10
        reconstr_loss = -tf.reduce_sum(self.x * tf.log(self.x_reconstr_mean), 1)  #/ tf.reduce_sum(self.x,1)
        latent_loss = 0.5*(tf.reduce_sum(tf.div(self.sigma, self.var2), 1) +
                           tf.reduce_sum(tf.multiply(tf.div((self.mu2 - self.z_mean), self.var2), (self.mu2 - self.z_mean)), 1) - \
                           self.n_z + tf.reduce_sum(tf.log(self.var2), 1) - tf.reduce_sum(self.z_log_sigma_sq, 1))
        self.cost = tf.reduce_mean(reconstr_loss) + tf.reduce_mean(latent_loss)  # average over batch
        if self.prior is not None:
            weight_recog = [self.network_weights["weights_enc"], self.network_weights["biases_enc"]]
            self.cost += tf.reduce_mean(self._prior_loss(self.prior, weight_recog))

        self.optimizer = \
            tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.99).minimize(self.cost)

    def _get_data_bin(self, data):
        data_boolean = tf.expand_dims(tf.greater(data, 0), 2)
        data_binarized = tf.logical_and(data_boolean, self.prior_bin)

        return tf.cast(data_binarized, dtype=tf.float32)

    def partial_fit(self, X, prior_bin):
        N, V = X.shape
        prior_bin = prior_bin[:N]
        opt, cost, beta = self.sess.run((self.optimizer, self.cost, 
                                        self.network_weights['weights_dec']['h1']), \
                                        feed_dict={self.x: X, self.keep_prob: 1.0 - self.dr, \
                                            self.lambda_: self.ld, self.bx: N, self.prior_bin: prior_bin})
        return cost, beta

    def gamma_test(self):
        """Test the model and return the lowerbound on the log-likelihood.
        """
        gamma = self.sess.run((self.Wo), feed_dict={self.keep_prob: 1.0, self.lambda_: self.ld})
        return gamma

    def test(self, X):
        """Test the model and return the lowerbound on the log-likelihood.
        """
        cost = self.sess.run((self.cost), feed_dict={self.x: X, self.keep_prob: 1.0, self.lambda_: self.ld})
        return cost

    def topic_prop(self, X):
        """Theta is the topic proportion vector. Apply softmax transformation to it before use.
        """
        theta = self.sess.run((tf.nn.softmax(self.z_mean)), \
                                feed_dict={self.x: X, self.keep_prob: 1.0, self.lambda_: self.ld})
        return theta
