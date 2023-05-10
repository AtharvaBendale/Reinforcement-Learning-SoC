import sys
import numpy as np

fileName = sys.argv[1]
loop = int(sys.argv[2]) if len(sys.argv)>2 else 200

class PV_algo:
    
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

    def pv_iteration(self):
        self.policy = np.zeros(self.states, dtype=int)
        self.vals = np.zeros(self.states)
        for iter in range(loop):
            for st in range(self.states):
                self.policy[st] = np.argmax(np.sum([[self.prob[st][i][j]*(self.r[st][i][j] + self.gm*self.vals[j]) for j in range(self.states)] for i in range(self.actions)], axis=1))
            for st in range(self.states):
                self.vals[st] = np.sum([self.prob[st][self.policy[st]][i]*(self.r[st][self.policy[st]][i] + self.gm*self.vals[i]) for i in range(self.states)])

    def store(self, out):
        with open(f"sol_{out}", 'w') as file:
            for i in range(self.states):
                file.write(f'{round(self.vals[i], 8)} {self.policy[i]}\n')

algo = PV_algo(fileName)
PV_algo.pv_iteration(algo)
PV_algo.store(algo, fileName)