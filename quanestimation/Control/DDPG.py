import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DDPG(object):
    def __init__(self, a_dim, s_dim, action_bound, actor_hidn_dim=200, critic_hidn_dim=200, \
                 actor_layer=2, critic_layer=1,activation='tanh', buffer_capacity=10000, \
                 batch_size=32, gamma=0.95, tau=0.02, actor_lr=0.001, critic_lr=0.002):
        """
        ---------
        Inputs
        ---------
        a_dim:
           --description: dimension of the action space.
           --type: int

        s_dim:
           --description: dimension of the state space.
           --type: int

        action_bound:
           --description: the boundary value of action.
           --type: float

        actor_hidn_dim:
           --description: hidden dimension of the actor network.
           --type: int

        critic_hidn_dim:
           --description: hidden dimension of the critic network.
           --type: int

        activation:
           --description: Activation function for the hidden layer.
                          'tanh' corresponds to the hyperbolic tan function with
                           the form f(x) = tanh(x);
                          'relu' corresponds to the rectified linear unit function
                          with the form f(x) = max(0, x);
                          'sigmoid' corresponds to logistic sigmoid function with
                           the form f(x) = 1 / (1+exp(-x)).
           --type: string {'tanh', 'relu', 'sigmoid'}

        buffer_capacity:
           --description: the capacity of the replay buffer.
           --type: int

        batch_size:
           --description: the number of data samples in one training step.
           --type: int

        gamma:
           --description: \gamma \in (0,1] is the reward discount rate.
           --type: float

        tau:
           --description: soft replacement factor.
           --type: float

        actor_lr:
           --description: learning rate for actor network.
           --type: float

        critic_lr:
           --description: learning rate for critic network.
           --type: float
        """
        self.a_dim, self.s_dim = a_dim, s_dim
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        self.memory = np.zeros((buffer_capacity, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.actor_eval = ddpg_actor(s_dim,a_dim,actor_hidn_dim,action_bound,actor_layer,activation)
        self.actor_target = ddpg_actor(s_dim,a_dim,actor_hidn_dim,action_bound,actor_layer,activation)
        self.critic_eval = ddpg_critic(s_dim,a_dim,critic_hidn_dim,critic_layer)
        self.critic_target = ddpg_critic(s_dim,a_dim,critic_hidn_dim,critic_layer)
        self.ctrain = torch.optim.Adam(self.critic_eval.parameters(),lr=critic_lr)
        self.atrain = torch.optim.Adam(self.actor_eval.parameters(),lr=actor_lr)
        self.loss_td = nn.MSELoss()

    def choose_action(self, s):
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        return self.actor_eval(s)[0].detach()

    def learn(self):
        for x in self.actor_target.state_dict().keys():
            eval('self.actor_target.' + x + '.data.mul_((1-self.tau))')
            eval('self.actor_target.' + x + '.data.add_(self.tau*self.actor_eval.' + x + '.data)')
        for x in self.critic_target.state_dict().keys():
            eval('self.critic_target.' + x + '.data.mul_((1-self.tau))')
            eval('self.critic_target.' + x + '.data.add_(self.tau*self.critic_eval.' + x + '.data)')

        indices = np.random.choice(self.buffer_capacity, size=self.batch_size)
        bt = self.memory[indices, :]
        bs = torch.FloatTensor(bt[:, :self.s_dim])
        ba = torch.FloatTensor(bt[:, self.s_dim: self.s_dim + self.a_dim])
        br = torch.FloatTensor(bt[:, -self.s_dim - 1: -self.s_dim])
        bs_ = torch.FloatTensor(bt[:, -self.s_dim:])

        a = self.actor_eval(bs)
        q = self.critic_eval(bs,a)
        loss_a = -torch.mean(q)
        self.atrain.zero_grad()
        loss_a.backward()
        self.atrain.step()

        a_ = self.actor_target(bs_)
        q_ = self.critic_target(bs_,a_)
        q_target = br+self.gamma*q_
        q_v = self.critic_eval(bs,ba)
        td_error = self.loss_td(q_target,q_v)

        self.ctrain.zero_grad()
        td_error.backward()
        self.ctrain.step()

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.buffer_capacity
        self.memory[index, :] = transition
        self.pointer += 1

class ddpg_actor(nn.Module):
    def __init__(self,s_dim,a_dim,actor_hidn_dim,action_bound,actor_layer,activation):

        super(ddpg_actor,self).__init__()
        self.action_bound = action_bound
        self.actor_layer = actor_layer
        self.activation = activation
        self.ent = nn.Linear(s_dim,actor_hidn_dim)
        self.ent.weight.data.normal_(0,0.1)

        self.fc = [nn.Linear(actor_hidn_dim,actor_hidn_dim) for i in range(actor_layer)]
        for j in range(actor_layer):
            self.fc[j].weight.data.normal_(0,0.1)

        self.out = nn.Linear(actor_hidn_dim,a_dim)
        self.out.weight.data.normal_(0,0.1)
    def forward(self,x_in):
        x1 = F.relu(self.ent(x_in))
        x = [x1 for m in range(self.actor_layer+1)]
        for n in range(self.actor_layer):
            x[n+1] = F.relu(self.fc[n](x[n]))
        x_out = self.out(x[-1])
        if self.activation == 'tanh':
            actions_value = self.action_bound*torch.tanh(x_out)
        elif self.activation == 'relu':
            actions_value = self.action_bound*F.relu(x_out)
        elif self.activation == 'sigmoid':
            actions_value = self.action_bound*torch.sigmoid(x_out)
        return actions_value

class ddpg_critic(nn.Module):
    def __init__(self,s_dim,a_dim,critic_hidn_dim,critic_layer):
        super(ddpg_critic,self).__init__()
        self.critic_layer = critic_layer
        self.s_ent = nn.Linear(s_dim,critic_hidn_dim)
        self.s_ent.weight.data.normal_(0,0.1)
        self.a_ent = nn.Linear(a_dim,critic_hidn_dim)
        self.a_ent.weight.data.normal_(0,0.1)

        self.fa = [nn.Linear(critic_hidn_dim,critic_hidn_dim) for i in range(critic_layer)]
        for j in range(critic_layer):
            self.fa[j].weight.data.normal_(0,0.1)

        self.out = nn.Linear(critic_hidn_dim,1)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self,s_in,a_in):
        x1 = self.s_ent(s_in)
        y1 = self.a_ent(a_in)
        net = F.relu(x1+y1)

        x = [net for m in range(self.critic_layer+1)]
        for n in range(self.critic_layer):
            x[n+1] = F.relu(self.fa[n](x[n]))

        actions_value = self.out(x[-1])
        return actions_value
