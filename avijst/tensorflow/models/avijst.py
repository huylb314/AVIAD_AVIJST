import numpy as np
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D
from tensorflow.contrib.keras.api.keras.initializers import Constant

def xavier_init(fan_in, fan_out, constant=1):
    low = -constant*tf.sqrt(6.0/(fan_in + fan_out))
    high = constant*tf.sqrt(6.0/(fan_in + fan_out))
    return tf.random.uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

tf.compat.v1.reset_default_graph()
class AVIJST(object):
    """
    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """


    def __init__(self, network_architecture, transfer_fct=tf.nn.softplus,
                 learning_rate=0.001, cls_learning_rate=0.005, batch_size=100, cls_model=None, save_file=None):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.cls_learning_rate = cls_learning_rate
        self.batch_size = batch_size
        print('Learning Rate: {0}, Classification Learning Rate: {1}'.format(self.learning_rate, self.cls_learning_rate))

        # tf Graph input
        self.x = tf.compat.v1.placeholder(tf.float32, [None, network_architecture["n_input"]])
        self.x_pi = tf.compat.v1.placeholder(tf.float32, [None, network_architecture["n_input_pi"]])
        self.y = tf.compat.v1.placeholder(tf.float32, [None, network_architecture["n_p"]])
        self.keep_prob = tf.compat.v1.placeholder(tf.float32)

        # n_z: number of aspect
        # n_p: number of sentiment
        self.h_dim = int(network_architecture["n_z"])
        self.pol_dim = int(network_architecture["n_p"])
        self.mix_dim = self.h_dim*self.pol_dim
        
        # alpha is SxK matrix: S distrution over K topic
        
        self.a = 1*np.ones((self.pol_dim , self.h_dim)).astype(np.float32)
        self.mu2 = tf.constant((np.log(self.a).T-np.mean(np.log(self.a),1)).T)
        self.var2 = tf.constant(  ( ( (1.0/self.a)*( 1 - (2.0/self.h_dim) ) ).T +
                                ( 1.0/(self.h_dim*self.h_dim) )*np.sum(1.0/self.a,1) ).T  )
        #"""
        self.g = 1*np.ones((1 , self.pol_dim)).astype(np.float32)
        self.pi_mu2 = tf.constant((np.log(self.g).T-np.mean(np.log(self.g),1)).T)
        self.pi_var2 = tf.constant(  ( ( (1.0/self.g)*( 1 - (2.0/self.pol_dim) ) ).T +
                                ( 1.0/(self.pol_dim*self.pol_dim) )*np.sum(1.0/self.g,1) ).T  )
        #"""
        """
        self.mu2 = tf.constant(np.zeros((self.pol_dim , self.h_dim)).astype(np.float32))
        self.var2 = tf.constant(np.ones((self.pol_dim , self.h_dim)).astype(np.float32))
        
        self.pi_mu2 = tf.constant(np.zeros((1 , self.pol_dim)).astype(np.float32))
        self.pi_var2 = tf.constant(np.ones((1 , self.pol_dim)).astype(np.float32))
        """

        self._create_network(cls_model)
        with tf.name_scope('cost'):
            self._create_loss_optimizer()
            self._create_cls_loss_optimizer()

        init = tf.compat.v1.global_variables_initializer()

        self.sess = tf.compat.v1.InteractiveSession()  
        self.saver = tf.compat.v1.train.Saver()
        
        # check exist checkpoint
        ckpt = None
        if save_file is not None:
            ckpt = tf.train.get_checkpoint_state(save_file)
        
        if ckpt:
            print ("Load checkpoint")
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print ("No checkpoint")
            self.sess.run(init)

    def _create_network(self, cls_model):
        self.network_weights = self._initialize_weights(**self.network_architecture)
        self.z_mean,self.z_log_sigma_sq,self.pi_mean,self.pi_log_sigma_sq = \
            self._recognition_network(self.network_weights["weights_recog"],
                                      self.network_weights["biases_recog"],
                                     cls_model)

        with tf.name_scope('z_sample'):
            eps = tf.random.normal((self.batch_size * self.pol_dim, self.h_dim), 0, 1,
                                   dtype=tf.float32)

            N, S, K = self.batch_size , self.pol_dim, self.h_dim
            z_mean = tf.reshape(self.z_mean, [N * S, K])
            z_sigma = tf.reshape(self.z_log_sigma_sq, [N * S, K])
            self.z = tf.add(z_mean,
                            tf.multiply(tf.sqrt(tf.exp(z_sigma)), eps))
            self.sigma = tf.exp(self.z_log_sigma_sq)

        with tf.name_scope('pi_sample'):
            pi_eps = tf.random.normal((self.batch_size, self.pol_dim), 0, 1,
                                   dtype=tf.float32)
            self.pi = tf.add(self.pi_mean,
                            tf.multiply(tf.sqrt(tf.exp(self.pi_log_sigma_sq)), pi_eps))
            self.pi_sigma = tf.exp(self.pi_log_sigma_sq)

        self.x_reconstr_mean = \
            self._generator_network(self.z,self.pi,self.network_weights["weights_gener"])

        print(self.x_reconstr_mean)

    def _initialize_weights(self, n_hidden_recog_1, n_hidden_recog_2,
                            n_hidden_gener_1,
                            n_input, n_input_pi, n_z, n_p):
        all_weights = dict()
        all_weights['weights_recog'] = {
            'h1': tf.compat.v1.get_variable('h1',[n_input, n_hidden_recog_1],
               initializer=tf.contrib.layers.xavier_initializer()),
            'h2': tf.compat.v1.get_variable('h2',[n_hidden_recog_1, n_hidden_recog_2],
               initializer=tf.contrib.layers.xavier_initializer()),
            'pi_h1': tf.compat.v1.get_variable('pi_h1',[n_input_pi, n_hidden_recog_1],
               initializer=tf.contrib.layers.xavier_initializer()),
            'pi_h2': tf.compat.v1.get_variable('pi_h2',[n_hidden_recog_1, n_hidden_recog_2],
               initializer=tf.contrib.layers.xavier_initializer()),
            'out_mean': tf.compat.v1.get_variable('out_mean',[n_hidden_recog_2, self.mix_dim],
               initializer=tf.contrib.layers.xavier_initializer()),
            'out_log_sigma': tf.compat.v1.get_variable('out_log_sigma',[n_hidden_recog_2, self.mix_dim],
               initializer=tf.contrib.layers.xavier_initializer()),
            'pi_out_mean': tf.compat.v1.get_variable('pi_out_mean',[n_hidden_recog_2, self.pol_dim],
               initializer=tf.contrib.layers.xavier_initializer()),
            'pi_out_log_sigma': tf.compat.v1.get_variable('pi_out_log_sigma',[n_hidden_recog_2, self.pol_dim],
               initializer=tf.contrib.layers.xavier_initializer()),
            }
        all_weights['biases_recog'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32), name='b1'),
            'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32), name='b2'),
            'pi_b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32), name='pi_b1'),
            'pi_b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32), name='pi_b2'),
            'out_mean': tf.Variable(tf.zeros([self.mix_dim], dtype=tf.float32), name='out_mean'),
            'out_log_sigma': tf.Variable(tf.zeros([self.mix_dim], dtype=tf.float32), name='out_log_sigma'),
            'pi_out_mean': tf.Variable(tf.zeros([self.pol_dim], dtype=tf.float32), name='pi_out_mean'),
            'pi_out_log_sigma': tf.Variable(tf.zeros([self.pol_dim], dtype=tf.float32), name='pi_out_log_sigma')}
        all_weights['weights_gener'] = {
            'h1': tf.Variable(xavier_init(self.pol_dim*self.h_dim, n_hidden_gener_1), name='gener_h1'),
            'h2': tf.Variable(xavier_init(self.pol_dim*self.h_dim,n_hidden_gener_1), name='gener_h2'),
            'h3': tf.Variable(xavier_init(self.pol_dim, n_hidden_gener_1), name='gener_h3')}

        return all_weights

    def _recognition_network(self, weights, biases, cls_model=None):
        max_features = self.network_architecture["n_input"]
        maxlen = self.network_architecture["n_input_pi"]
        embedding_dims = 100
        filters = 250
        kernel_size = 3
        hidden_dims = self.network_architecture['n_hidden_recog_1']#500
        
        # Generate probabilistic encoder (recognition network)
        with tf.name_scope('z_layers'):
            layer_1 = self.transfer_fct(tf.add(tf.matmul(self.x, weights['h1']),
                                               biases['b1']))
            layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']),
                                               biases['b2']))
            layer_do = tf.nn.dropout(layer_2, rate=1 - self.keep_prob)

        N, S, K = self.batch_size , self.pol_dim, self.h_dim
        with tf.name_scope('z_mean'):
            layer_dout_mean = tf.add(tf.matmul(layer_do, weights['out_mean']),
                            biases['out_mean'])
            layer_dout_mean = tf.reshape(layer_dout_mean, [N, S, K])
            z_mean = tf.contrib.layers.batch_norm(layer_dout_mean)

        with tf.name_scope('z_sigma'):
            layer_dout_sigma = tf.add(tf.matmul(layer_do, weights['out_log_sigma']),
                       biases['out_log_sigma'])
            layer_dout_sigma = tf.reshape(layer_dout_sigma, [N, S, K])
            z_log_sigma_sq = tf.contrib.layers.batch_norm(layer_dout_sigma)

        with tf.name_scope('pi_layers'):
            if cls_model is None:
                pi_layer = Embedding(max_features, embedding_dims, input_length=maxlen)(self.x_pi)
                pi_layer = Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1)(pi_layer)
                pi_layer = GlobalMaxPooling1D()(pi_layer)
                pi_layer = Dense(hidden_dims)(pi_layer)
                pi_layer = Activation('relu')(pi_layer)
            else:
                pi_layer = cls_model(self.x_pi)
            
            pi_layer_do = tf.nn.dropout(pi_layer, rate=1 - self.keep_prob)#Dropout(0.4)(pi_layer)
            
        with tf.name_scope('pi_mean'):
            pi_mean = tf.contrib.layers.batch_norm(tf.add(tf.matmul(pi_layer_do, weights['pi_out_mean']),
                            biases['pi_out_mean']))
            #pi_mean = Dense(self.pol_dim)(pi_layer_do)
            #pi_mean = BatchNormalization()(pi_mean)
        
        with tf.name_scope('pi_sigma'):
            pi_log_sigma_sq = \
                tf.contrib.layers.batch_norm(tf.add(tf.matmul(pi_layer_do, weights['pi_out_log_sigma']),
                       biases['pi_out_log_sigma']))        
            #pi_log_sigma_sq = Dense(self.pol_dim)(pi_layer_do)
            #pi_log_sigma_sq = BatchNormalization()(pi_log_sigma_sq)
        
        return (z_mean, z_log_sigma_sq, pi_mean, pi_log_sigma_sq)

    def _generator_network(self,z, pi, weights):
        N, S, K = self.batch_size , self.pol_dim, self.h_dim
        # z is (NxS)xK
        z_distr = tf.reshape(z, [N, S, K])
        self.layer_do_z = tf.nn.softmax(tf.contrib.layers.batch_norm(z_distr))
        self.layer_do_s = tf.nn.softmax(tf.contrib.layers.batch_norm(pi))
        self.lay_s = 0.001*tf.matmul(self.layer_do_s, weights['h3'])
        #self.lay_s = 0.0*tf.matmul(self.layer_do_s, weights['h3'])
        
        s = tf.reshape(self.layer_do_s, [N, S, 1])
        s = tf.tile(s, [1, 1, K])
        with tf.name_scope('Decoder'):
            # Nx(S*K) * (NxS).repeat(K) * (S*K)xD
            sz = tf.reshape(self.layer_do_z*s, [N, S*K])
            fc = tf.add(tf.matmul(sz, weights['h2']), self.lay_s)
        with tf.name_scope('x_reconstr'):
            x_reconstr_mean = tf.nn.softmax(tf.contrib.layers.batch_norm(fc))

        return x_reconstr_mean

    def _create_loss_optimizer(self):

        self.x_reconstr_mean+=1e-10

        reconstr_loss = \
            -tf.reduce_sum(self.x * tf.math.log(self.x_reconstr_mean),1)#/tf.reduce_sum(self.x,1)

        latent_z_loss = 0.5*( tf.reduce_sum(tf.math.divide(self.sigma,self.var2),-1)+\
        tf.reduce_sum( tf.multiply(tf.math.divide((self.mu2 - self.z_mean),self.var2),
                  (self.mu2 - self.z_mean)),-1) - self.h_dim +\
                           tf.reduce_sum(tf.math.log(self.var2),-1)  - tf.reduce_sum(self.z_log_sigma_sq  ,-1) )

        latent_s_loss = 0.5*( tf.reduce_sum(tf.math.divide(self.pi_sigma,self.pi_var2),1)+\
        tf.reduce_sum( tf.multiply(tf.math.divide((self.pi_mu2 - self.pi_mean),self.pi_var2),
                  (self.pi_mu2 - self.pi_mean)),1) - self.pol_dim +\
                           tf.reduce_sum(tf.math.log(self.pi_var2),1)  - tf.reduce_sum(self.pi_log_sigma_sq  ,1) )
        

        self.kl_s_loss = tf.reduce_mean(latent_s_loss)
        self.kl_z_loss = tf.reduce_sum(tf.reduce_mean(latent_z_loss,0),-1) 
        self.kl_loss =  self.kl_s_loss + self.kl_z_loss
        self.cost = tf.reduce_mean(reconstr_loss) + self.kl_loss# average over batch

        self.optimizer = \
            tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate,beta1=0.99).minimize(self.cost)

    def _create_cls_loss_optimizer(self):
        self.logits = tf.nn.softmax(self.pi_mean)
        self.cls_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.logits, labels=self.y))
        #self.cls_loss = tf.reduce_mean(categorical_crossentropy(self.y, self.logits))
        self.cls_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.cls_learning_rate).minimize(self.cls_loss)

    def cls_fit(self, X, Y):
        #K.set_learning_phase(1)
        opt, cost = self.sess.run((self.cls_optimizer, self.cls_loss),feed_dict={self.x_pi: X, self.y: Y, self.keep_prob: .4})        
        return cost    

    def partial_fit(self, X, X_pi):
        opt, cost, kl_s_loss, kl_z_loss, emb = self.sess.run((self.optimizer, self.cost, self.kl_s_loss, self.kl_z_loss ,self.network_weights['weights_gener']['h2']),feed_dict={self.x: X, self.x_pi: X_pi,self.keep_prob: .4})
        
        return cost,kl_s_loss,kl_z_loss,emb

    def get_weights(self):
        gen_w = self.network_weights['weights_gener']
        h2, h3 = self.sess.run((gen_w['h2'], gen_w['h3']))
        return [h2, h3]
    
    def test(self, X):
        """Test the model and return the lowerbound on the log-likelihood.
        """
        cost = self.sess.run((self.cost),feed_dict={self.x: np.expand_dims(X, axis=0),self.keep_prob: 1.0})
        return cost
    
    def topic_prop(self, X):
        """theta_ is the topic proportion vector. Apply softmax transformation to it before use.
        """
        N, S, K = self.batch_size , self.pol_dim, self.h_dim
        z_mean = tf.reshape(self.z_mean, [N * S, K])
        theta_ = self.sess.run(tf.nn.softmax(z_mean),feed_dict={self.x: X,self.keep_prob: 1.0})
        return theta_

    def senti_prop(self, X):
        K.set_learning_phase(0)
        """pi_ is the sentiment proportion vector. Apply softmax transformation to it before use.
        """
        pi_ = self.sess.run(self.logits,feed_dict={self.x_pi: X,self.keep_prob: 1.0})
        return pi_
    
    def save_model(self, save_file, epoch_numb):
        # save model
        self.saver.save(self.sess, save_file, global_step=epoch_numb)