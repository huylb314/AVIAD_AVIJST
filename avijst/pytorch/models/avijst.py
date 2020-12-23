import torch
import torch.nn as nn
import torch.nn.functional as F

class AVIJST(nn.Module):
    def __init__(self, num_input, en1_units, en2_units, \
                 num_topic, num_sentiment, drop_rate, \
                 init_mult, num_layers, bidirectional, pad_idx):
        super(ProdLDA, self).__init__()
        self.num_input, self.en1_units, self.en2_units, \
        self.num_topic, self.num_sentiment, self.drop_rate, \
        self.init_mult, self.num_layers, self.bidirectional, self.pad_idx = \
                num_input, en1_units, en2_units, \
                num_topic, num_sentiment, drop_rate, \
                init_mult, num_layers, bidirectional, pad_idx
        self.num_mix = self.num_topic * self.num_sentiment
        self.batch_first, self.enforce_sorted = True, False
                
        ## PI
        # encoder: pi
        # self.pi_en1_fc = nn.Linear(self.num_input, self.en1_units)
        # self.pi_en1_ac = nn.Softplus()
        # self.pi_en2_fc     = nn.Linear(self.en1_units, self.en2_units)
        # self.pi_en2_ac = nn.Softplus()
        # self.pi_en2_dr   = nn.Dropout(self.drop_rate)
        # sentiment classification: pi 
        self.pi_emb = nn.Embedding(self.num_input, self.en1_units, padding_idx=self.pad_idx)
        self.pi_rnn = nn.LSTM(self.en1_units, self.en2_units, num_layers=self.num_layers, \
                              bidirectional=self.bidirectional, batch_first=self.batch_first)
        self.pi_fc = nn.Linear(self.en2_units * 2, self.en2_units)
        self.pi_dr = nn.Dropout(self.drop_rate)
        self.pi_ac = nn.ReLU()
        # mean, logvar: pi
        self.pi_mean_fc = nn.Linear(self.en2_units, self.num_sentiment)
        self.pi_mean_bn = nn.BatchNorm1d(self.num_sentiment)
        self.pi_logvar_fc = nn.Linear(self.en2_units, self.num_sentiment)
        self.pi_logvar_bn = nn.BatchNorm1d(self.num_sentiment)
        # decoder: pi
        self.pi_de_ac = nn.Softmax(dim=-1)
        self.pi_de_fc = nn.Linear(self.num_sentiment, self.num_input)
        # prior mean and variance: pi (1 x S) S_sentiments
        self.pi_prior_mean   = torch.Tensor(1, self.num_sentiment).fill_(0)
        self.pi_prior_var    = torch.Tensor(1, self.num_sentiment).fill_(0.995)
        self.pi_prior_mean   = nn.Parameter(self.pi_prior_mean, requires_grad=False)
        self.pi_prior_var    = nn.Parameter(self.pi_prior_var, requires_grad=False)
        self.pi_prior_logvar = nn.Parameter(self.pi_prior_var.log(), requires_grad=False)
        
        ## THETA
        # encoder: theta
        self.theta_en1_fc = nn.Linear(self.num_input, self.en1_units)
        self.theta_en1_ac = nn.Softplus()
        self.theta_en2_fc     = nn.Linear(self.en1_units, self.en2_units)
        self.theta_en2_ac = nn.Softplus()
        self.theta_en2_dr   = nn.Dropout(self.drop_rate)
        # mean, logvar: theta
        self.theta_mean_fc = nn.Linear(self.en2_units, self.num_mix)
        self.theta_mean_bn = nn.BatchNorm1d(self.num_mix)
        self.theta_logvar_fc = nn.Linear(self.en2_units, self.num_mix)
        self.theta_logvar_bn = nn.BatchNorm1d(self.num_mix)
        # decoder: theta
        self.theta_de_ac = nn.Softmax(dim=-1)
        # prior mean and variance: theta (S x K) (S_sentiments) distribution over (K_topics)
        self.theta_prior_mean = torch.Tensor(self.num_sentiment, self.num_topic).fill_(0)
        self.theta_prior_var    = torch.Tensor(self.num_sentiment, self.num_topic).fill_(0.995)
        self.theta_prior_mean   = nn.Parameter(self.theta_prior_mean, requires_grad=False)
        self.theta_prior_var    = nn.Parameter(self.theta_prior_var, requires_grad=False)
        self.theta_prior_logvar = nn.Parameter(self.theta_prior_var.log(), requires_grad=False)
        
        # mix theta pi
        self.theta_pi_de_fc = nn.Linear(self.num_mix, self.num_input)
        self.theta_pi_de_bn = nn.BatchNorm1d(self.num_input)
        self.theta_pi_de_ac = nn.Softmax(dim=-1)
        
        # initialize decoder weight
        if init_mult != 0:
            #std = 1. / math.sqrt( init_mult * (num_topic + num_input))
            self.pi_de_fc.weight.data.uniform_(0, init_mult)
            self.theta_pi_de_fc.weight.data.uniform_(0, init_mult)
        # remove BN's scale parameters
        for component in [self.pi_mean_bn, self.pi_logvar_bn, \
                          self.theta_mean_bn, self.theta_logvar_bn, self.theta_pi_de_bn]:
            component.weight.requires_grad = False
            component.weight.fill_(1.0)
            
    def encode(self, input_, input_r_, input_len_):
        # extract bs
        bs, *_ = input_.size()
        
        ## THETA
        # encoder: theta
        theta_encoded1 = self.theta_en1_fc(input_)
        theta_encoded1_ac = self.theta_en1_ac(theta_encoded1)
        theta_encoded2 = self.theta_en2_fc(theta_encoded1_ac)
        theta_encoded2_ac = self.theta_en2_ac(theta_encoded2)
        theta_encoded2_dr = self.theta_en2_dr(theta_encoded2_ac)
        theta_encoded = theta_encoded2_dr
        # hidden => mean, logvar: theta
        theta_mean = self.theta_mean_fc(theta_encoded)
        theta_mean_bn = self.theta_mean_bn(theta_mean)
        theta_mean_reshaped = theta_mean_bn.view(bs, self.num_sentiment, self.num_topic)
        theta_logvar = self.theta_logvar_fc(theta_encoded)
        theta_logvar_bn = self.theta_logvar_bn(theta_logvar)
        theta_logvar_reshaped = theta_logvar_bn.view(bs, self.num_sentiment, self.num_topic)
        # posterior: theta
        theta_posterior_mean = theta_mean_reshaped # N x S x K
        theta_posterior_logvar = theta_logvar_reshaped # N x S x K
        
        ## PI
        # encoder: pi
        # pi_encoded1 = self.pi_en1_fc(input_)
        # pi_encoded1_ac = self.pi_en1_ac(pi_encoded1)
        # pi_encoded2 = self.pi_en2_fc(pi_encoded1_ac)
        # pi_encoded2_ac = self.pi_en2_ac(pi_encoded2_ac)
        # pi_encoded2_dr = self.pi_en2_dr(pi_encoded2_dr)
        # pi_encoded = pi_encoded2_dr
        # encoder: pi_cls 
        pi_embedded = self.pi_dr(self.pi_emb(input_r_))
        # pack sequence
        pi_packed_embedded = nn.utils.rnn.pack_padded_sequence(pi_embedded, input_len_, batch_first=self.batch_first, enforce_sorted=self.enforce_sorted)
        #embedded = [bs, sent_len, emb_dim]
        pi_packed_output, (pi_hidden_rnn, pi_cell_rnn) = self.pi_rnn(pi_packed_embedded)
        #unpack sequence
        pi_output, pi_output_len_ = nn.utils.rnn.pad_packed_sequence(pi_packed_output, batch_first=self.batch_first)
        #output = [bs, sent_len, hid dim * num directions]
        #output over padding tokens are zero tensors
        #hidden = [bs, num layers * num directions, hid dim]
        #cell = [bs, num layers * num directions, hid dim]
        #concat the final forward (hidden[:,-2,:]) and backward (hidden[:,-1,:]) hidden layers
        #and apply dropout
        pi_hidden = self.pi_dr(torch.cat((pi_hidden_rnn[-2, :, :], pi_hidden_rnn[-1, :, :]), dim = 1))
        pi_hidden_fc = self.pi_fc(pi_hidden)
        pi_hidden_ac = self.pi_ac(pi_hidden_fc)
        pi_encoded = pi_hidden_ac

        # hidden => mean, logvar: pi
        pi_mean = self.pi_mean_fc(pi_encoded)
        pi_mean_bn = self.pi_mean_bn(pi_mean)
        pi_logvar = self.pi_logvar_fc(pi_encoded)
        pi_logvar_bn = self.pi_logvar_bn(pi_logvar)
        # posterior: pi
        pi_posterior_mean = pi_mean_bn # N x S
        pi_posterior_logvar = pi_logvar_bn # N x S
        
        return theta_encoded, pi_encoded, \
                theta_posterior_mean, theta_posterior_logvar, \
                pi_posterior_mean, pi_posterior_logvar
    
    def decode(self, input_, theta_posterior_mean, theta_posterior_var, pi_posterior_mean, pi_posterior_var):
        bs, *_ = input_.size()
        # take sample: theta
        theta_eps = input_.data.new().resize_as_(theta_posterior_mean.data).normal_() # noise 
        theta_eps_reshaped = theta_eps.view(bs * self.num_sentiment, self.num_topic)
        theta_posterior_mean_reshaped = theta_posterior_mean.view(bs * self.num_sentiment, self.num_topic)
        theta_posterior_var_reshaped = theta_posterior_var.view(bs * self.num_sentiment, self.num_topic)
        # reparameterization
        theta_posterior = theta_posterior_mean_reshaped + theta_posterior_var_reshaped.sqrt() * theta_eps_reshaped                   
        theta_posterior_reshaped = theta_posterior.view(bs, self.num_sentiment, self.num_topic)
        theta_posterior_ac = self.theta_de_ac(theta_posterior_reshaped)
        
        # take sample: pi
        pi_lambda = 0.001
        pi_eps = input_.data.new().resize_as_(pi_posterior_mean.data).normal_() # noise 
        # reparameterization
        pi_posterior = pi_posterior_mean + pi_posterior_var.sqrt() * pi_eps 
        pi_posterior_ac = self.pi_de_ac(pi_posterior)
        pi_decoded = pi_lambda * self.pi_de_fc(pi_posterior_ac) # N x V
        pi_posterior_reshaped = pi_posterior_ac.view(bs, self.num_sentiment, 1) # N x S x 1
        pi_posterior_repeated = pi_posterior_reshaped.repeat((1, 1, self.num_topic)) # N x S x K
        
        # Nx(S*K) * (NxS).repeat(K) * (S*K)xD
        # NxSxK * NxSxK
        theta_pi_mix = theta_posterior_ac * pi_posterior_repeated
        theta_pi_mix_reshaped = theta_pi_mix.view(bs, self.num_mix)
        
        # do reconstruction
        theta_pi_decoded = self.theta_pi_de_fc(theta_pi_mix_reshaped)
        theta_pi_decoded_bn = self.theta_pi_de_bn(theta_pi_decoded)
        theta_pi_decoded_ac = self.theta_pi_de_ac(theta_pi_decoded_bn)
        recon = theta_pi_decoded_ac          # reconstructed distribution over vocabulary
        
        return recon
    
    def forward(self, input_, input_r_, input_len_, compute_loss=False, avg_loss=True):
        # compute posterior
        theta_encoded, pi_encoded, \
        theta_posterior_mean, theta_posterior_logvar, \
        pi_posterior_mean, pi_posterior_logvar = self.encode(input_, input_r_, input_len_) 
        theta_posterior_var    = theta_posterior_logvar.exp()
        pi_posterior_var = pi_posterior_logvar.exp()
        
        recon = self.decode(input_, theta_posterior_mean, theta_posterior_var, \
                                    pi_posterior_mean, pi_posterior_var )
        if compute_loss:
            return recon, self.loss(input_, recon, \
                                    theta_posterior_mean, theta_posterior_logvar, theta_posterior_var, \
                                    pi_posterior_mean, pi_posterior_logvar, pi_posterior_var, \
                                    avg_loss)
        else:
            return recon, theta_encoded, pi_encoded, \
                    theta_posterior_mean, theta_posterior_logvar, theta_posterior_var, \
                     pi_posterior_mean, pi_posterior_logvar, pi_posterior_var
    
    def loss(self, input_, recon, \
             theta_posterior_mean, theta_posterior_logvar, theta_posterior_var, \
             pi_posterior_mean, pi_posterior_logvar, pi_posterior_var, \
             avg=True):
        # NL
        NL  = -(input_ * (recon + 1e-10).log()).sum(1)
        # KLD, see Section 3.3 of Akash Srivastava and Charles Sutton, 2017, 
        # https://arxiv.org/pdf/1703.01488.pdf
        # sentiments: pi
        pi_prior_mean   = self.pi_prior_mean.expand_as(pi_posterior_mean)
        pi_prior_var    = self.pi_prior_var.expand_as(pi_posterior_mean)
        pi_prior_logvar = self.pi_prior_logvar.expand_as(pi_posterior_mean)
        pi_var_division    = pi_posterior_var  / pi_prior_var
        pi_diff            = pi_posterior_mean - pi_prior_mean
        pi_diff_term       = pi_diff * pi_diff / pi_prior_var
        pi_logvar_division = pi_prior_logvar - pi_posterior_logvar
        # put KLD together
        pi_KLD = 0.5 * ( (pi_var_division + pi_diff_term + pi_logvar_division).sum(1) - self.num_sentiment)
        
        # topics: theta
        theta_prior_mean   = self.theta_prior_mean.expand_as(theta_posterior_mean) # N x S x K
        theta_prior_var    = self.theta_prior_var.expand_as(theta_posterior_mean)
        theta_prior_logvar = self.theta_prior_logvar.expand_as(theta_posterior_mean)
        theta_var_division    = theta_posterior_var  / theta_prior_var
        theta_diff            = theta_posterior_mean - theta_prior_mean
        theta_diff_term       = theta_diff * theta_diff / theta_prior_var
        theta_logvar_division = theta_prior_logvar - theta_posterior_logvar
        # put KLD together
        theta_KLD = 0.5 * ( (theta_var_division + theta_diff_term + theta_logvar_division).sum(-1) - self.num_topic)
        
        # loss
        loss = (NL.mean() + pi_KLD.mean() + theta_KLD.mean(0).sum(-1))
        
        # in traiming mode, return averaged loss. In testing mode, return individual loss
        if avg:
            return loss
        else:
            return loss