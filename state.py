"""
0 = unknown
1 = less
2 = equal
3 = greater
"""
class State: 
    """Stores the state of a list of n objects. In particular, stores whether the value 
    of that index is greater than, less than, or equal to a particular key.

    Attributes:
    - key:       an integer key with which to compare list elmeents, should be in [0, n-1]
    - n:         The length of the list to be stored
    - A:         a list of the integers [0,...,n-1]
    - num_steps: the number of queries already preformed
    - bad_step:  is true if you try to check an index that has already been checked
    """
    def __init__(self, n, key):
        self.key = key
        self.A = [0 for _ in range(n)]
        self.n = n
        self.num_steps = 0
        self.bad_step = False

    def _check_index(self, idx):
        less = self.key < idx
        greater = self.key > idx
        eq = self.key == idx
        new = self.A[idx] == 0
        if self.A[idx] != 0:
            self.bad_step = True
        if less:
            self.A[idx] = -1
        elif greater:
            self.A[idx] = 1
        return eq, new

    def step(self, idx):
        # Make it possible for the model to declare that the key is not in the list.
        if (idx == self.n):
            done = True
            if (self.key < self.n):
                reward = -10
            return reward, done
        
        eq, new = self._check_index(idx)
        reward = -0.1 - 10 * self.bad_step + 0.05 * new
        done = False
        self.num_steps += 1

        self.bad_step = False
        if eq:
            done = True 
            reward += 10
        # Make sure that the episode eventually terminates
        if self.num_steps > 0.4 * self.n:
            reward -= 10
            done = True
        return reward, done
    
    def reset(self, key, n):
        self.key = key
        self.A = [0 for _ in range(n)]
        self.n = n
        self.num_steps = 0
        self.bad_step = False
