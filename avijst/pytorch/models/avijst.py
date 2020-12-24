import torch
import torch.nn as nn
import torch.nn.functional as F

class AVIJST(nn.Module):
    def __init__(self, num_input, en1_units, en2_units, num_topic, num_senti, dr_topic, dr_senti, init_mult):
        super(AVIJST, self).__init__()
        self.num_input, self.en1_units, self.en2_units, \
            self.num_topic, self.num_senti, \
                self.dr_topic, self.dr_senti, \
                    self.init_mult = num_input, en1_units, en2_units, \
                                        num_topic, num_senti, \
                                            dr_topic, dr_senti, \
                                                init_mult

        # encoder: theta
        self.en1_fc_theta     = nn.Linear(self.num_input, self.en1_units) # N x V => N x 500
        self.en2_fc_theta     = nn.Linear(self.en1_units, self.en2_units)
        self.en2_drop_theta   = nn.Dropout(self.dr_topic)
        # hidden
        self.mean_fc_theta    = nn.Linear(self.en2_units, self.num_topic * self.num_senti)
        self.mean_bn_theta    = nn.BatchNorm1d(self.num_senti * self.num_topic)           
        self.logvar_fc_theta  = nn.Linear(self.en2_units, self.num_topic * self.num_senti)
        self.logvar_bn_theta  = nn.BatchNorm1d(self.num_senti * self.num_topic)
        # decoder
        self.p_drop_theta     = nn.Dropout(self.dr_topic)
        self.de_fc_theta    = nn.Linear(self.num_topic * self.num_senti, self.num_input, bias=False)
        self.de_bn1_theta = nn.BatchNorm1d(self.num_senti * self.num_topic)
        self.de_bn2_theta = nn.BatchNorm1d(self.num_input)
        
        # encoder: pi MLP
        self.en1_fc_pi     = nn.Linear(self.num_input, self.en1_units) # N x V => N x 500
        self.en2_fc_pi     = nn.Linear(self.en1_units, self.en2_units)
        self.en2_drop_pi   = nn.Dropout(self.dr_senti)
        # hidden
        self.mean_fc_pi    = nn.Linear(self.en2_units, self.num_senti)
        self.mean_bn_pi    = nn.BatchNorm1d(self.num_senti)           
        self.logvar_fc_pi  = nn.Linear(self.en2_units, self.num_senti)
        self.logvar_bn_pi  = nn.BatchNorm1d(self.num_senti)   
        # decoder
        self.de_bn_pi = nn.BatchNorm1d(self.num_senti)
        self.de_fc_pi = nn.Linear(self.num_senti, self.num_input, bias=False)
        
        # prior mean and variance as constant buffers
        prior_mean_theta   = torch.Tensor(1, self.num_senti, self.num_topic).fill_(0)
        prior_var_theta    = torch.Tensor(1, self.num_senti, self.num_topic).fill_(0.995)
        self.prior_mean_theta   = nn.Parameter(prior_mean_theta, requires_grad=False)
        self.prior_var_theta    = nn.Parameter(prior_var_theta, requires_grad=False)
        self.prior_logvar_theta = nn.Parameter(prior_var_theta.log(), requires_grad=False)
        
        prior_mean_pi   = torch.Tensor(1, self.num_senti).fill_(0)
        prior_var_pi    = torch.Tensor(1, self.num_senti).fill_(0.995)
        self.prior_mean_pi   = nn.Parameter(prior_mean_pi, requires_grad=False)
        self.prior_var_pi    = nn.Parameter(prior_var_pi, requires_grad=False)
        self.prior_logvar_pi = nn.Parameter(prior_var_pi.log(), requires_grad=False)
        
        # initialize decoder weight
        if init_mult != 0:
            #std = 1. / math.sqrt( init_mult * (num_topic + num_input))
            self.de_fc_pi.weight.data.uniform_(0, self.init_mult, )
            self.de_fc_theta.weight.data.uniform_(0, self.init_mult)
        # remove BN's scale parameters
        for component in [self.mean_bn_theta, self.logvar_bn_theta, \
                          self.mean_bn_pi, self.logvar_bn_pi]:
            component.weight.requires_grad = False
            component.weight.fill_(1.0)
        for component in [self.de_bn1_theta, self.de_bn2_theta, self.de_bn_pi]:
            component.weight.requires_grad = False
            component.weight.fill_(1.0)
               
    def encode(self, bs, input_, input_r_, input_len_):
        # encoder: theta
        en1_theta = F.softplus(self.en1_fc_theta(input_))
        en2_theta = F.softplus(self.en2_fc_theta(en1_theta)) 
        encoded_theta = self.en2_drop_theta(en2_theta) # N x (S * K)
        # hidden
        posterior_mean_theta_fc = self.mean_fc_theta  (encoded_theta)
        posterior_mean_theta_reshaped   = posterior_mean_theta_fc.view(bs, self.num_senti * self.num_topic)  # N x S x K
        posterior_mean_theta   = self.mean_bn_theta  (posterior_mean_theta_reshaped)       # N x S x K
        posterior_mean_theta   = posterior_mean_theta.view(bs, self.num_senti, self.num_topic)  # N x S x K
        posterior_logvar_theta_fc = self.logvar_fc_theta(encoded_theta)
        posterior_logvar_theta_reshaped = posterior_logvar_theta_fc.view(bs, self.num_senti * self.num_topic)# N x S x K
        posterior_logvar_theta = self.logvar_bn_theta(posterior_logvar_theta_reshaped)     # N x S x K
        posterior_logvar_theta = posterior_logvar_theta.view(bs, self.num_senti, self.num_topic)# N x S x K
        
        # encoder: pi 
        en1_pi = F.softplus(self.en1_fc_pi(input_)) # N x S
        en2_pi = F.softplus(self.en2_fc_pi(en1_pi)) # N x S
        encoded_pi = self.en2_drop_pi(en2_pi)       # N x S
        # hidden
        posterior_mean_pi   = self.mean_bn_pi  (self.mean_fc_pi  (encoded_pi)) # N x S
        posterior_logvar_pi = self.logvar_bn_pi(self.logvar_fc_pi(encoded_pi)) # N x S
        
        return posterior_mean_theta, posterior_logvar_theta, posterior_logvar_theta.exp(), \
                posterior_mean_pi, posterior_logvar_pi, posterior_logvar_pi.exp()
    
    def take_sample(self, input_, posterior_mean_theta, posterior_var_theta, posterior_mean_pi, posterior_var_pi):
        # take sample: theta
        eps_theta = input_.data.new().resize_as_(posterior_mean_theta.data).normal_() # noise N x S x K
        posterior_theta = posterior_mean_theta + posterior_var_theta.sqrt() * eps_theta #  N x S x K
        
        # take sample: pi
        eps_pi = input_.data.new().resize_as_(posterior_mean_pi.data).normal_() # noise N x S
        posterior_pi = posterior_mean_pi + posterior_var_pi.sqrt() * eps_pi # N x S
        
        return posterior_theta, posterior_pi
    
    def decode(self, bs, posterior_theta, posterior_pi):        
        # posterior_theta:  N x S x K
        # posterior_pi: N x S
        # z_distr = tf.reshape(z, [N, S, K])
        # self.layer_do_z = tf.nn.softmax(tf.contrib.layers.batch_norm(z_distr))
        posterior_theta_reshaped = posterior_theta.view(bs, self.num_senti * self.num_topic)
        layer_do_posterior_theta_bn = self.de_bn1_theta(posterior_theta_reshaped)
        layer_do_posterior_theta_bn_reshaped = \
                layer_do_posterior_theta_bn.view(bs, self.num_senti, self.num_topic)
        layer_do_posterior_theta_ac = F.softmax(layer_do_posterior_theta_bn_reshaped, dim=-1)
        layer_do_posterior_theta = layer_do_posterior_theta_ac # N x S x K
        
        layer_do_posterior_pi_bn = self.de_bn_pi(posterior_pi)
        layer_do_posterior_pi_ac = F.softmax(layer_do_posterior_pi_bn, dim=-1)
        layer_do_posterior_pi = layer_do_posterior_pi_ac # N x S
        
        layer_pi = 0.01 * self.de_fc_pi(layer_do_posterior_pi) # N x V
        
        layer_do_posterior_pi_reshaped = layer_do_posterior_pi.view(bs, self.num_senti, 1) # N x S x 1
        layer_do_posterior_pi_repeated = layer_do_posterior_pi_reshaped.repeat((1, 1, self.num_topic)) # N x S x K
        
        theta_pi = layer_do_posterior_theta * layer_do_posterior_pi_repeated
        theta_pi_reshaped = theta_pi.view(bs, self.num_senti * self.num_topic) # N x (S * K)
        theta_pi_layer_pi = self.de_fc_theta(theta_pi_reshaped) + layer_pi # N x V
        
        theta_pi_bn = self.de_bn2_theta(theta_pi_layer_pi)
        theta_pi_ac = F.softmax(theta_pi_bn, dim=-1)
        
        # do reconstruction
        recon = theta_pi_ac
        return recon
    
    def forward(self, input_r_, input_len_, input_,  label_, cls_loss=False, compute_loss=False, avg_loss=True):
        bs, *_ = input_.size()
        # compute posterior
        posterior_mean_theta, posterior_logvar_theta, posterior_var_theta,\
            posterior_mean_pi, posterior_logvar_pi, posterior_var_pi = self.encode(bs, input_, input_r_, input_len_) 
        
        # take sample
        posterior_theta, posterior_pi = self.take_sample(input_, \
                                                         posterior_mean_theta, posterior_var_theta, \
                                                         posterior_mean_pi, posterior_var_pi)
        
        recon = self.decode(bs, posterior_theta, posterior_pi)
        if compute_loss:
            if cls_loss:
                return recon, self.loss_cls(posterior_mean_pi, posterior_logvar_pi, posterior_var_pi, label_, avg_loss)
            else:
                return recon, self.loss(bs, input_, recon, \
                                        posterior_mean_theta, posterior_logvar_theta, posterior_var_theta, \
                                        posterior_mean_pi, posterior_logvar_pi, posterior_var_pi, avg_loss)
        else:
            return recon, posterior_mean_theta, posterior_logvar_theta, posterior_mean_pi, posterior_logvar_pi

    def loss_cls(self, posterior_mean_pi, posterior_logvar_pi, posterior_var_pi, label, avg=True):
        posterior_mean_pi_sm = F.softmax(posterior_mean_pi, dim=-1)
        loss_l = F.cross_entropy(posterior_mean_pi_sm, label)
        if avg:
            return loss_l.mean()
        else:
            return loss_l
        
        
    def loss(self, bs, input_, recon, \
             posterior_mean_theta, posterior_logvar_theta, posterior_var_theta, \
             posterior_mean_pi, posterior_logvar_pi, posterior_var_pi, \
             avg=True):
        # NL
        NL  = -(input_ * (recon + 1e-10).log()).sum(1)
        
        posterior_mean_theta = posterior_mean_theta.view(bs, self.num_senti, self.num_topic)
        posterior_logvar_theta = posterior_logvar_theta.view(bs, self.num_senti, self.num_topic)
        posterior_var_theta = posterior_var_theta.view(bs, self.num_senti, self.num_topic)
        # KLD, see Section 3.3 of Akash Srivastava and Charles Sutton, 2017, 
        # https://arxiv.org/pdf/1703.01488.pdf
        prior_mean_theta   = self.prior_mean_theta.expand_as(posterior_mean_theta) # N x S x K
        prior_var_theta    = self.prior_var_theta.expand_as(posterior_mean_theta) 
        prior_logvar_theta = self.prior_logvar_theta.expand_as(posterior_mean_theta)
        var_division_theta    = posterior_var_theta  / prior_var_theta
        diff_theta            = posterior_mean_theta - prior_mean_theta
        diff_term_theta       = diff_theta * diff_theta / prior_var_theta
        logvar_division_theta = prior_logvar_theta - posterior_logvar_theta
        # put KLD together
        KLD_theta = 0.5 * ( (var_division_theta + diff_term_theta + logvar_division_theta).sum(-1) - self.num_topic)
        
        prior_mean_pi   = self.prior_mean_pi.expand_as(posterior_mean_pi) # N x S
        prior_var_pi    = self.prior_var_pi.expand_as(posterior_mean_pi)
        prior_logvar_pi = self.prior_logvar_pi.expand_as(posterior_mean_pi)
        var_division_pi    = posterior_var_pi  / prior_var_pi
        diff_pi            = posterior_mean_pi - prior_mean_pi
        diff_term_pi       = diff_pi * diff_pi / prior_var_pi
        logvar_division_pi = prior_logvar_pi - posterior_logvar_pi
        # put KLD together
        KLD_pi = 0.5 * ( (var_division_pi + diff_term_pi + logvar_division_pi).sum(1) - self.num_senti)
        
        #self.kl_loss =  tf.reduce_mean(latent_s_loss) + tf.reduce_sum(tf.reduce_mean(latent_z_loss,0),-1) 
        #self.cost = tf.reduce_mean(reconstr_loss) + self.kl_loss# average over batch

        # in traiming mode, return averaged loss. In testing mode, return individual loss
        if avg:
            kl_loss = KLD_theta.sum(-1) + KLD_pi
            # loss
            loss = (NL.mean() + kl_loss.mean())
            return loss
        else:
            return loss