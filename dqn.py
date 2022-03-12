from collections import deque
import matplotlib
# matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.autograd as autograd
import math, random
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class QLearner(nn.Module):
    def __init__(self, env, num_frames, batch_size, gamma, replay_buffer):
        super(QLearner, self).__init__()

        self.batch_size = batch_size
        self.gamma = gamma
        self.num_frames = num_frames
        self.replay_buffer = replay_buffer
        self.env = env
        self.input_shape = self.env.observation_space.shape
        self.num_actions = self.env.action_space.n

        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def feature_size(self):
            return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), requires_grad=True)
            # TODO: Given state, you should write code to get the Q value and chosen action

            # run the state through the model, model will output list of action values, 
            # we need to pick action that gives max value
            # use model(state) and then torch.argmax()
        
            action_list = self(state)
            action = torch.argmax(action_list)
            return action

        else:
            action = random.randrange(self.env.action_space.n)
            return action

    def copy_from(self, target):
        self.load_state_dict(target.state_dict())

        
def compute_td_loss(model, target_model, batch_size, gamma, replay_buffer):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = Variable(torch.FloatTensor(np.float32(state)).squeeze(1))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)).squeeze(1), requires_grad=True)
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))
    # implement the loss function here

    # use model to get actual q value, target_model to get expected
    # target_model(next_state), model(state)
    # run the state through the model, model will output list of action values,
    # Q_curr = Qs_curr[range(batch_size), action.cpu()] : picks the action value from the state

    # Run the state through the model to get actual value q, which is a vector of n values
    q_curr = model(state).gather(1, action.unsqueeze(-1)).squeeze(-1)

    # Get the q value of the next action using target model
    q_next = target_model(next_state).detach().max(1)[0]
    q_next[(done == 1)] = 0

    # Get the expected values y_j
    y_j = reward + (gamma * q_next)

    # Get the sum of loss: sum(y_j - q_curr)
    loss = torch.sum( (y_j - q_curr)**2 )
 
    return loss


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # TODO: Randomly sampling data with specific batch size from the buffer

        # Sample list just be a list of tuples of size 5: (state, action, reward, next_state, done) python unzip command

        # we take random sample from batch_size
        sample_list = random.sample(self.buffer, batch_size)

        # Unzip the list of tuples
        sample_list_unzip = list(zip(*sample_list))

        # Variables will be lists since 
        state = sample_list_unzip[0]
        action = sample_list_unzip[1]
        reward = sample_list_unzip[2]
        next_state = sample_list_unzip[3]
        done = sample_list_unzip[4]
        
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
