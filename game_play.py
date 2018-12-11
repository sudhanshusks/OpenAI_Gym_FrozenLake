from __future__ import division
import gym
from gym.envs.registration import register
import numpy as np
import random, math, time
import copy
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

register(
    id          ='FrozenLakeNotSlippery-v0',
    entry_point ='gym.envs.toy_text:FrozenLakeEnv',
    kwargs      ={'map_name' : '8x8', 'is_slippery': False},
)

def running_mean(x, N=20):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

class Agent:
    def __init__(self, env):
        self.stateCnt      = env.observation_space.n
        self.actionCnt     = env.action_space.n         # left:0; down:1; right:2; up:3
        self.learning_rate = 0.8
        self.gamma         = 0.9
        self.epsilon       = 0.1
        self.Q             = self._initialiseModel()

    def _initialiseModel(self):
        qtable= np.zeros((self.stateCnt, self.actionCnt))
        return qtable

    def predict_value(self, s):
        action= self.Q[s, :]
        return action

    def update_value_Qlearning(self, s,a,r,s_next, goalNotReached):
        if(goalNotReached):
            predict= self.Q[s, a]
            target= r+ self.gamma*np.max(self.Q[s_next, :])
            self.Q[s, a]= self.Q[s, a]+ self.learning_rate*(target- predict)
        else:
            self.Q[s, a]= self.Q[s, a]+ self.learning_rate*(r - self.Q[s, a])

    def update_value_SARSA(self, s,a,r,s_next, a_next, goalNotReached):
        if(goalNotReached):
            predict= self.Q[s, a]
            target= r+ self.gamma*self.Q[s_next, a_next]
            self.Q[s, a]= self.Q[s, a]+ self.learning_rate*(target- predict)
        else:
            self.Q[s, a]= self.Q[s, a]+ self.learning_rate*(r - self.Q[s, a])

    def choose_action(self, s):
        if(np.random.uniform(0, 1) < self.epsilon):
            best_action= env.action_space.sample()
        else:
            best_action= np.argmax(self.Q[s, :]+ np.random.randn(1, 4))
        return best_action

    def updateEpsilon(self, episodeCounter):
        #self.epsilon= 0.01+ (1- 0.01)*np.exp(-0.005*episodeCounter)
        self.epsilon-= .001
        return

class World:
    def __init__(self, env):
        self.env = env
        print('Environment has %d states and %d actions.' % (self.env.observation_space.n, self.env.action_space.n))
        self.stateCnt           = self.env.observation_space.n
        self.actionCnt          = self.env.action_space.n
        self.maxStepsPerEpisode = 100
        self.q_Sinit_progress   = np.array([[0, 0, 0, 0]])   # ex: np.array([[0,0,0,0]])

    def run_episode_qlearning(self):
        s               = self.env.reset() # "reset" environment to start state
        r_total         = 0
        episodeStepsCnt = 0
        success         = False
        for i in range(self.maxStepsPerEpisode):
            # self.env.step(a): "step" will execute action "a" at the current agent state and move the agent to the next state.
            # step will return the next state, the reward, a boolean indicating if a terminal state is reached,
            #and some diagnostic information useful for debugging.
            
            # self.env.render(): "render" will print the current environment state.
            #self.env.render()
            action= agent.choose_action(s)
            s_next, r, done, info= self.env.step(action)
            agent.update_value_Qlearning(s, action, r, s_next, not done)
            r_total+= r
            s= s_next
            #print(r_total)
            if(done):   #we are dead, finish episode
                #self.env.render()
                #print("No of steps :", i+1)
                episodeStepsCnt= i+1
                break
            # self.q_Sinit_progress = np.append( ): use q_Sinit_progress for monitoring the q value progress 
            #throughout training episodes for all available actions at the initial state.
        self.q_Sinit_progress= np.append(self.q_Sinit_progress, [agent.Q[s, :]], axis=0)
        return r_total, episodeStepsCnt

    def run_episode_sarsa(self):
        s               = self.env.reset() # "reset" environment to start state
        r_total         = 0
        episodeStepsCnt = 0
        success         = False
        action= agent.choose_action(s)
        for i in range(self.maxStepsPerEpisode):
            # self.env.step(a): "step" will execute action "a" at the current agent state and move the agent to the next state.
            # step will return the next state, the reward, a boolean indicating if a terminal state is reached, 
            #and some diagnostic information useful for debugging.
            
            # self.env.render(): "render" will print the current environment state.
            #self.env.render
            s_next, r, done, info= self.env.step(action)
            action2= agent.choose_action(s_next)
            agent.update_value_SARSA(s, action, r, s_next, action2, not done)
            r_total+= r
            s= s_next
            action= action2
            if(done):
                #self.env.render()
                #print("No of steps :", i+1)
                episodeStepsCnt= i+1
                break
            # self.q_Sinit_progress = np.append( ): use q_Sinit_progress for monitoring the q value progress 
            #throughout training episodes for all available actions at the initial state
        self.q_Sinit_progress= np.append(self.q_Sinit_progress, [agent.Q[s, :]], axis=0)
        return r_total, episodeStepsCnt

    def run_evaluation_episode(self):
        agent.epsilon = 0
        state= self.env.reset()
        step= 0
        success= False
        for step in range(self.maxStepsPerEpisode):
            self.env.render()
            action= np.argmax(agent.Q[state, :])
            new_state, reward, done, info= self.env.step(action)
            state= new_state
            if(reward == 1):
                success= True
                break    
        return success


if __name__ == '__main__':
    env                      = gym.make('FrozenLakeNotSlippery-v0')
    world                    = World(env)
    agent                    = Agent(env) # This will creat an agent
    r_total_progress         = []
    episodeStepsCnt_progress = []
    nbOfTrainingEpisodes     = 1000
    for i in range(nbOfTrainingEpisodes):
        print ('\n========================\n   Episode: {}\n========================'.format(i))
        # run_episode_qlearning or run_episode_sarsa
        r_total_tmp, episodeStepsCnt_tmp= world.run_episode_qlearning()
        #r_total_tmp, episodeStepsCnt_tmp= world.run_episode_sarsa()
        agent.updateEpsilon(i)
        print(r_total_tmp)
        # append to r_total_progress and episodeStepsCnt_progress
        r_total_progress.append(r_total_tmp)
        episodeStepsCnt_progress.append(episodeStepsCnt_tmp)
    # run_evaluation_episode
    print(world.run_evaluation_episode())
    
    ### --- Plots --- ###
    # 1) plot world.q_Sinit_progress
    fig1 = plt.figure(1)
    plt.ion()
    plt.plot(world.q_Sinit_progress[:,0], label='left',  color = 'r')
    plt.plot(world.q_Sinit_progress[:,1], label='down',  color = 'g')
    plt.plot(world.q_Sinit_progress[:,2], label='right', color = 'b')
    plt.plot(world.q_Sinit_progress[:,3], label='up',    color = 'y')
    fontP = FontProperties()
    fontP.set_size('small')
    plt.legend(prop = fontP, loc=1)
    plt.pause(0.001)

    # 2) plot the evolution of the number of steps per successful episode throughout training. A successful episode is an episode where the agent reached the goal (i.e. not any terminal state)
    fig2 = plt.figure(2)
    plt1 = plt.subplot(1,2,1)
    plt1.set_title("Number of steps per successful episode")
    plt.ion()
    plt.plot(episodeStepsCnt_progress)
    plt.pause(0.0001)
    # 3) plot the evolution of the total collected rewards per episode throughout training. you can use the running_mean function to smooth the plot
    plt2 = plt.subplot(1,2,2)
    plt2.set_title("Rewards collected per episode")
    plt.ion()
    r_total_progress = running_mean(r_total_progress)
    plt.plot(r_total_progress)
    plt.pause(0.0001)
    ### --- ///// --- ###








