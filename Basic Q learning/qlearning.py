import gym
import numpy as np

env=gym.make("MountainCar-v0")

# env.reset()

print(env.observation_space.high)
print(env.observation_space.low)
print(env.action_space.n) 

#[0.6,0.07]
#[-1.2,-0.07]

#20*2 =[20,20]
#0.6 is velocity, 0.7 is position
discrete_observation_size=[20]*len(env.observation_space.high)

#create 20 equal spaces from 0.6 to -1.2 and 0.07 to -0.07
#[1.8,0.14]/[20,20]=[0.09,0.007]
discrete_observation_spaces=(env.observation_space.high - env.observation_space.low)/discrete_observation_size 

#create a qtable
q_table=np.random.uniform(low=-2,high=0,size=(discrete_observation_size+[env.action_space.n]))

learning_rate=0.1
discount=0.95
episodes=25000

def get_discrete_state(state):
    discrete_state=(state - env.observation_space.low)/discrete_observation_spaces
    return tuple(discrete_state.astype(np.int64))



epsilon=1
start_epsilon_decaying=1
end_epsilon_decaying=episodes//2
epsilon_decaying_rate=epsilon//(end_epsilon_decaying-start_epsilon_decaying)


for episode in range(episodes):
    discrete_state=get_discrete_state(env.reset())
    done=False
    while not done:

        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(q_table[discrete_state])
        else:
            # Get random action
            action = np.random.randint(0, env.action_space.n)
    
        new_state,reward,done,_=env.step(action)

        if episode%2000==0:
            env.render()
            print(episode)

        new_discrete_state=get_discrete_state(new_state)

        if not done:
            max_future_q=np.max(q_table[discrete_state])
            current_q=q_table[discrete_state+(action,)]
            new_q=(1-learning_rate)*current_q+learning_rate*(reward+discount+max_future_q)
            q_table[discrete_state+(action,)]=new_q

        elif new_state[0]>env.goal_position:
            #q_table[discrete_state + (action,)] = reward
            q_table[discrete_state + (action,)] = 0
            print(done)

        discrete_state=new_discrete_state
        
    if end_epsilon_decaying >= episode >= start_epsilon_decaying:
        epsilon -= epsilon_decaying_rate

env.close()
