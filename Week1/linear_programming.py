import numpy as np
import sys

fileName = sys.argv[1]
loop = int(sys.argv[2]) if len(sys.argv)>2 else 200

class LP_algo:
    def __init__(self, fileName):
        with open(f"MDPs/{fileName}") as file:
            for line in file:
                keys = line.split(" ")
                if keys[0] == "states":
                    self.states = int(keys[1])
                elif keys[0] == "actions":
                    self.actions = int(keys[1])
                    self.prob = np.zeros((self.states, self.actions, self.states))
                    self.r = np.zeros((self.states, self.actions, self.states))
                elif keys[0] == "tran":
                    self.prob[int(keys[1])][int(keys[2])][int(keys[3])] = float(keys[5][:-1])
                    self.r[int(keys[1])][int(keys[2])][int(keys[3])] = float(keys[4])
                elif keys[0] == "gamma":
                    self.gm = float(keys[1])
    
    def my_lp(self):
        self.vals = np.ones(self.states, dtype=float)
        self.policy = np.ones(self.states, dtype=int)
        for i1 in range(loop):
            for i2 in range(self.states):
                self.vals[i2] = np.max(np.array([np.sum([ self.prob[i2][i][j]*(self.r[i2][i][j] + self.gm*self.vals[j]) for j in range(self.states)]) for i in range(self.actions)]))
        for st in range(self.states):
            self.policy[st] = np.argmax(np.sum([[self.prob[st][i][j]*(self.r[st][i][j] + self.gm*self.vals[j]) for j in range(self.states)] for i in range(self.actions)], axis=1))

    def store(self, out):
        with open(f"sol_{out}", 'w') as file:
            for i in range(self.states):
                file.write(f'{round(self.vals[i], 8)} {self.policy[i]}\n')

algo = LP_algo(fileName)
LP_algo.my_lp(algo)
LP_algo.store(algo, fileName)