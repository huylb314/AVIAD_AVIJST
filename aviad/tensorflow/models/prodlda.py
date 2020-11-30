import numpy as np
import tensorflow as tf


def xavier_init(fan_in, fan_out, constant=1):
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)


def log_dir_init(fan_in, fan_out, topics=50):
    return tf.log((1.0/topics)*tf.ones([fan_in, fan_out]))


tf.reset_default_graph()


class ProdLDA(object):
    """
    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """

    def __init__(self, n_hidden_recog_1, n_hidden_recog_2, n_hidden_gener_1,
                 n_input, n_z, data_prior, transfer_fct=tf.nn.softplus,
                 learning_rate=0.001, batch_size=100):
        self.n_hidden_recog_1 = n_hidden_recog_1
        self.n_hidden_recog_2 = n_hidden_recog_2
        self.n_hidden_gener_1 = n_hidden_gener_1
        self.n_input = n_input
        self.n_z = n_z
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # tf Graph input
        self.prior, prior_bin = data_prior
        self.x = tf.placeholder(tf.float32, [None, n_input])
        self.x_bin = self._get_data_bin(self.x, self.prior, prior_bin)
        self.keep_prob = tf.placeholder(tf.float32)

        self.h_dim = n_z
        self.a = 1*np.ones((1, self.h_dim)).astype(np.float32)
        self.mu2 = tf.constant((np.log(self.a).T-np.mean(np.log(self.a), 1)).T)
        self.var2 = tf.constant((((1.0/self.a)*(1 - (2.0/self.h_dim))).T +
                                 (1.0/(self.h_dim*self.h_dim))*np.sum(1.0/self.a, 1)).T)

        self._create_network()
        self._create_loss_optimizer()

        init = tf.initialize_all_variables()

        self.sess = tf.InteractiveSession()
        self.sess.run(init)

    def _create_network(self):
        self.network_weights = self._initialize_weights(
            self.n_hidden_recog_1, self.n_hidden_recog_2,
            self.n_hidden_gener_1, self.n_input, self.n_z
        )
        self.z_mean, self.z_log_sigma_sq = \
            self._recognition_network(self.network_weights["weights_recog"],
                                      self.network_weights["biases_recog"])

        eps = tf.random_normal((self.batch_size, self.n_z), 0, 1,
                               dtype=tf.float32)
        self.z = tf.add(self.z_mean,
                        tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))
        self.sigma = tf.exp(self.z_log_sigma_sq)

        self.x_reconstr_mean = \
            self._generator_network(
                self.z, self.network_weights["weights_gener"])

    def _initialize_weights(self, n_hidden_recog_1, n_hidden_recog_2,
                            n_hidden_gener_1,
                            n_input, n_z):
        all_weights = dict()
        all_weights['weights_recog'] = {
            'h1': tf.get_variable('h1', [n_input, n_hidden_recog_1],
                                  initializer=tf.contrib.layers.xavier_initializer()),
            'h2': tf.get_variable('h2', [n_hidden_recog_1, n_hidden_recog_2],
                                  initializer=tf.contrib.layers.xavier_initializer()),
            'out_mean': tf.get_variable('out_mean', [n_hidden_recog_2, n_z],
                                        initializer=tf.contrib.layers.xavier_initializer()),
            'out_log_sigma': tf.get_variable('out_log_sigma', [n_hidden_recog_2, n_z],
                                             initializer=tf.contrib.layers.xavier_initializer())}
        all_weights['biases_recog'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}
        all_weights['weights_gener'] = {
            'h2': tf.Variable(xavier_init(n_z, n_hidden_gener_1))}

        return all_weights

    def _recognition_network(self, weights, biases):
        # Generate probabilistic encoder (recognition network)
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.x, weights['h1']),
                                           biases['b1']))
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']),
                                           biases['b2']))
        layer_do = tf.nn.dropout(layer_2, self.keep_prob)

        z_mean = tf.contrib.layers.batch_norm(tf.add(tf.matmul(layer_do, weights['out_mean']),
                                                     biases['out_mean']))
        z_log_sigma_sq = \
            tf.contrib.layers.batch_norm(tf.add(tf.matmul(layer_do, weights['out_log_sigma']),
                                                biases['out_log_sigma']))

        return (z_mean, z_log_sigma_sq)

    def _generator_network(self, z, weights):
        self.layer_do_0 = tf.nn.dropout(tf.nn.softmax(z), self.keep_prob)
        x_reconstr_mean = tf.nn.softmax(tf.contrib.layers.batch_norm(tf.add(
            tf.matmul(self.layer_do_0, weights['h2']), 0.0)))
        return x_reconstr_mean

    def _prior_loss(self, prior, weights):
        self.prior_loss = 0

        W = weights[0]
        b = weights[1]
        W1 = self.transfer_fct(tf.add(W['h1'], b['b1']))
        W2 = self.transfer_fct(tf.add(tf.matmul(W1, W['h2']), b['b2']))
        Wdr = tf.nn.dropout(W2, self.keep_prob)
        Wo = tf.nn.softmax(tf.contrib.layers.batch_norm(
            tf.add(tf.matmul(Wdr, W['out_mean']), b['out_mean'])))
        self.Wo = Wo

        self.prior_loss = 20.0 * \
            tf.reduce_sum((prior - self.x_bin*Wo)**2, [1, 2])
        return self.prior_loss

    def _create_loss_optimizer(self):
        self.x_reconstr_mean += 1e-10

        reconstr_loss = \
            -tf.reduce_sum(self.x * tf.log(self.x_reconstr_mean),
                           1)  # /tf.reduce_sum(self.x,1)

        latent_loss = 0.5*(tf.reduce_sum(tf.div(self.sigma, self.var2), 1) +
                           tf.reduce_sum(tf.multiply(tf.div((self.mu2 - self.z_mean), self.var2),
                                                     (self.mu2 - self.z_mean)), 1) - self.h_dim +
                           tf.reduce_sum(tf.log(self.var2), 1) - tf.reduce_sum(self.z_log_sigma_sq, 1))

        self.cost = tf.reduce_mean(
            reconstr_loss) + tf.reduce_mean(latent_loss)  # average over batch
        if self.prior is not None:
            weight_recog = [self.network_weights["weights_recog"],
                            self.network_weights["biases_recog"]]
            self.cost += tf.reduce_mean(
                self._prior_loss(self.prior, weight_recog))

        self.optimizer = \
            tf.train.AdamOptimizer(
                learning_rate=self.learning_rate, beta1=0.99).minimize(self.cost)

    def _get_data_bin(self, data, gamma, gamma_bin):
        data_bool = tf.expand_dims(tf.greater(data, 0), 2)
        data_bin = tf.logical_and(data_bool, gamma_bin)

        return tf.cast(data_bin, dtype=tf.float32)

    def partial_fit(self, X):
        opt, cost, emb = self.sess.run((self.optimizer, self.cost, self.network_weights['weights_gener']['h2']), feed_dict={
                                       self.x: X, self.keep_prob: .4})
        return cost, emb

    def gamma_test(self):
        """Test the model and return the lowerbound on the log-likelihood.
        """
        gamma = self.sess.run((self.Wo), feed_dict={self.keep_prob: 1.0})
        return gamma

    def test(self, X):
        """Test the model and return the lowerbound on the log-likelihood.
        """
        cost = self.sess.run((self.cost), feed_dict={
                             self.x: X, self.keep_prob: 1.0})
        return cost

    def topic_prop(self, X):
        """Theta_ is the topic proportion vector. Apply softmax transformation to it before use.
        """
        theta_ = self.sess.run((tf.nn.softmax(self.z_mean)), feed_dict={
                               self.x: X, self.keep_prob: 1.0})
        return theta_
