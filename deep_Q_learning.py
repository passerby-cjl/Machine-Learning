import random
import time
import gym
import numpy as np
import deep_neural_network as DNN

class DQNModel(DNN.DNNModel):
    def __init__(self, gamma:float, batch_size, epsilon, S_l:list, learning_rate:float, lamda:int=0, theta = None):
        DNN.DNNModel.__init__(self, S_l, learning_rate, lamda, theta)
        self.D=[]
        self.greedy=1.0
        self.batch_size=batch_size
        self.epsilon=epsilon
        self.gamma = gamma

    def trainDQN(self, myset):
        # cur_states=[]
        # next_states=[]
        # for id in range(len(myset)):
        #     _cur_state, _action, _r, _next_state, _done=myset[id]
        #     next_states.append(_next_state)
        #     cur_states.append(_cur_state)
        cur_states, _a,_b,next_states, _c=zip(*myset)
        cur_states=np.array(cur_states)
        next_states=np.array(next_states)
        next_states_q=self.hypothesis(next_states)[-1]
        cur_states_q=self.hypothesis(cur_states)[-1]
        for id in range(len(myset)):
            _cur_state, _action, _r, _next_state, _done=myset[id]
            if not _done:
                cur_states_q[id][_action]=float(_r+self.gamma*np.max(next_states_q[id]))
            else:
                cur_states_q[id][_action]=float(_r)
        self.train(cur_states, cur_states_q)

    def get_best_action(self, _state, prt=False):
        if random.random() < self.greedy:
            return random.randint(0,self.K-1)
        else:
            actions=self.hypothesis(list(_state))[-1]
            if prt:
                print(actions, np.argmax(actions))
            return int(np.argmax(actions))

if __name__ == "__main__":
    env=gym.make("MountainCar-v1")#v0的最大步数改为10000
    test_model=DQNModel(0.9, 100, 0.01, [(3,""),(24,"ReLU"),(24,"ReLU"),(4,"Liner")], 0.001)
    counts=[]
    for times in range(5000):
        count=0
        state=env.reset()
        if times >= 10 and test_model.greedy>test_model.epsilon:
            test_model.greedy *= 0.995
        while True:
            count+=1
            action=test_model.get_best_action(state)
            next_state, reward, done, _=env.step([action-2])
            if not done or next_state[0]>=0.5:
                test_model.D.append((list(state), action, reward, list(next_state), done))
            if times >= 10:
                sets = random.sample(test_model.D, test_model.batch_size)
                test_model.trainDQN(sets)
            state=next_state
            if done:
                counts.append(count)
                print(count)
                if len(counts)>10:
                    counts=counts[1:]
                if np.mean(counts) <= 160:
                    state = env.reset()
                    done=False
                    while not done:
                        env.render()
                        state,reward,done,_=env.step(test_model.get_best_action(state, True))
                        time.sleep(0.2)
                    #env.close()
                break
    state=env.reset()
    done=False
    while not done:
        env.render()
        state,reward,done,_=env.step(test_model.get_best_action(state, True))
        time.sleep(0.02)
    env.close()

