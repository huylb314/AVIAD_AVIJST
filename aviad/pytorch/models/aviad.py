import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AVIAD(nn.Module):
    def __init__(self, n_hidden_enc_1, n_hidden_enc_2, n_vocab, n_topic, gamma_prior, gammar_prior_bin, init_mult=1.0, variance=0.995, \
                        ld=20.0, al=1.0, dr=0.6):
        super(AVIAD, self).__init__()
        self.H1, self.H2 = n_hidden_enc_1, n_hidden_enc_2
        self.V = n_vocab
        self.K = n_topic

        self.init_mult = init_mult
        self.variance = variance

        self.ld = ld # lambda
        self.al = al # alpha 
        self.dr = dr # dropout

        # gamma prior
        self.gamma_prior = gamma_prior  
        self.gammar_prior_bin = gammar_prior_bin
        if torch.cuda.is_available():
            self.gamma_prior = gamma_prior.cuda()
            self.gammar_prior_bin = gammar_prior_bin.cuda()
        
        # encoder
        self.en1_fc = nn.Linear(self.V, self.H1)
        self.en1_ac = nn.Softplus()
        self.en2_fc = nn.Linear(self.H1, self.H2)
        self.en2_ac = nn.Softplus()
        self.en2_dr = nn.Dropout(self.dr)
        
        # mean, logvar
        self.mean_fc = nn.Linear(self.H2, self.K)
        self.mean_bn = nn.BatchNorm1d(self.K)
        self.logvar_fc = nn.Linear(self.H2, self.K)
        self.logvar_bn = nn.BatchNorm1d(self.K)

        # decoder
        self.de_ac1 = nn.Softmax(dim=-1) # NxK 
        self.de_dr = nn.Dropout(self.dr)
        self.de_fc = nn.Linear(self.K, self.V)
        self.de_bn = nn.BatchNorm1d(self.V)
        self.de_ac2 = nn.Softmax(dim=-1) # NxV
        
        # prior mean and variance as constant buffers
        self.prior_mean   = torch.Tensor(1, self.K).fill_(0)
        self.prior_var    = torch.Tensor(1, self.K).fill_(self.variance)
        self.prior_mean   = nn.Parameter(self.prior_mean, requires_grad=False)
        self.prior_var    = nn.Parameter(self.prior_var, requires_grad=False)
        self.prior_logvar = nn.Parameter(self.prior_var.log(), requires_grad=False)
        # initialize decoder weight
        if self.init_mult != 0:
            # std = 1. / math.sqrt( init_mult * (self.K + self.V))
            self.de_fc.weight.data.uniform_(0, self.init_mult)
        # remove BN's scale parameters
        for component in [self.mean_bn, self.logvar_bn, self.de_bn]:
            component.weight.requires_grad = False
            component.weight.fill_(1.0)
        
    def gamma_test(self):
        # this function have to run after self.encode
        encoder_w1 = self.en1_fc.weight
        encoder_b1 = self.en1_fc.bias
        encoder_w2 = self.en2_fc.weight
        encoder_b2 = self.en2_fc.bias
        mean_w = self.mean_fc.weight
        mean_b = self.mean_fc.bias
        mean_running_mean = self.mean_bn.running_mean
        mean_running_var = self.mean_bn.running_var
        logvar_w = self.logvar_fc.weight
        logvar_b = self.logvar_fc.bias
        logvar_running_mean = self.logvar_bn.running_mean
        logvar_running_var = self.logvar_bn.running_var
        
        w1 = F.softplus(encoder_w1.t() + encoder_b1)
        w2 = F.softplus(F.linear(w1, encoder_w2, encoder_b2))
        wdr = F.dropout(w2, self.dr)
        wo_mean = F.softmax(F.linear(wdr, mean_w, mean_b), dim=-1)
        # wo_mean = F.softmax(F.batch_norm(F.linear(wdr, mean_w, mean_b), mean_running_mean, mean_running_var), dim=-1)
        
        return wo_mean
            
    def encode(self, x):
        # encoder
        encoded1 = self.en1_fc(x)
        encoded1_ac = self.en1_ac(encoded1)
        encoded2 = self.en2_fc(encoded1_ac)
        encoded2_ac = self.en2_ac(encoded2)
        encoded2_dr = self.en2_dr(encoded2_ac)
        
        encoded = encoded2_dr
        
        # hidden => mean, logvar
        mean_theta = self.mean_fc(encoded)
        mean_theta_bn = self.mean_bn(mean_theta)
        logvar_theta = self.logvar_fc(encoded)
        logvar_theta_bn = self.logvar_bn(logvar_theta)
        
        posterior_mean = mean_theta_bn
        posterior_logvar = logvar_theta_bn
        return encoded, posterior_mean, posterior_logvar
    
    def decode(self, x, posterior_mean, posterior_var):
        # take sample
        eps = x.data.new().resize_as_(posterior_mean.data).normal_() # noise 
        z = posterior_mean + posterior_var.sqrt() * eps                   # reparameterization
        # do reconstruction
        # decoder
        decoded1_ac = self.de_ac1(z)
        decoded1_dr = self.de_dr(decoded1_ac)
        decoded2 = self.de_fc(decoded1_dr)
        decoded2_bn = self.de_bn(decoded2)
        decoded2_ac = self.de_ac2(decoded2_bn)
        recon = decoded2_ac          # reconstructed distribution over vocabulary
        return recon
    
    def forward(self, x, avg_loss=True):
        # compute posterior
        en2, posterior_mean, posterior_logvar = self.encode(x) 
        posterior_var    = posterior_logvar.exp()
        
        recon = self.decode(x, posterior_mean, posterior_var)
        return self.loss(x, recon, posterior_mean, posterior_logvar, posterior_var, avg_loss), self.de_fc.weight.data.cpu().numpy().T

    def loss(self, x, recon, posterior_mean, posterior_logvar, posterior_var, avg=True):
        # NL
        NL  = -(x * (recon + 1e-10).log()).sum(1)
        # KLD, see Section 3.3 of Akash Srivastava and Charles Sutton, 2017, 
        # https://arxiv.org/pdf/1703.01488.pdf
        prior_mean   = self.prior_mean.expand_as(posterior_mean)
        prior_var    = self.prior_var.expand_as(posterior_mean)
        prior_logvar = self.prior_logvar.expand_as(posterior_mean)
        var_division    = posterior_var  / prior_var
        diff            = posterior_mean - prior_mean
        diff_term       = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar
        # put KLD together
        KLD = 0.5 * ( (var_division + diff_term + logvar_division).sum(1) - self.K)
        
        # gamma
        N, _ = x.size()
        gamma_mean = self.gamma_test()
        gamma_prior, gammar_prior_bin = self.gamma_prior, self.gammar_prior_bin
        x_boolean = (x > 0).unsqueeze(dim=-1) # NxVx1
        x_gamma_boolean = ((gammar_prior_bin[:N, :, :].expand(N, -1, -1) == 1) & x_boolean)
        lambda_c = self.ld
        
        gamma_prior = gamma_prior.expand(N, -1, -1)      
        
        GL = lambda_c * ((gamma_prior - (x_gamma_boolean.int()*gamma_mean))**2).sum((1, 2))
        
        # loss
        loss = (NL + KLD + GL)
        
        # in traiming mode, return averaged loss. In testing mode, return individual loss
        if avg:
            return loss.mean()
        else:
            return loss