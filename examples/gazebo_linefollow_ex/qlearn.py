import random
import pickle
import csv


class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    def loadQ(self, filename):
        '''
        Load the Q state-action values from a pickle file.
        '''
        # TODO: Implement loading Q values from pickle file.
        path= os.path.join(os.getcwd(), filename+".pickle")
        print(f"Loading Q values from: {path}")

        try:
            with open(filename+".pickle", 'rb') as file:
                self.q= pickle.load(file)
            print("Loaded pickle file")
        except Exception as e:
            print("Error loading:")


    def saveQ(self, filename):
        '''
        Save the Q state-action values in a pickle file.
        '''
        # TODO: Implement saving Q values to pickle and CSV files.

        with open(filename+".pickle", 'wb') as picklefile:
            pickle.dump(self.q, picklefile)
        print("Wrote to pickle file:")
        
        with open(filename+".csv", 'w', newline='') as csvfile:
            writer= csv.writer(csvfile)
            for key,value in self.q.items():
                state,action=key
                writer.writerow([state,action,value])
        print("Wrote to csv file")

    def getQ(self, state, action):
        '''
        @brief returns the state, action Q value or 0.0 if the value is 
            missing
        '''
        return self.q.get((state, action), 0.0)

    def chooseAction(self, state, return_q=False):
        '''
        @brief returns a random action epsilon % of the time or the action 
            associated with the largest Q value in (1-epsilon)% of the time
        '''
        # TODO: Implement exploration vs exploitation
        #    if we need to take a random action:
        #       * return a random action
        #    else:
        #       * determine which action has the highest Q value for the state 
        #          we are in.
        #       * address edge cases - what if 2 actions have the same max Q 
        #          value?
        #       * return the action with highest Q value
        #
        # NOTE: if return_q is set to True return (action, q) instead of
        #       just action

        # THE NEXT LINES NEED TO BE MODIFIED TO MATCH THE REQUIREMENTS ABOVE 
        
        #Explore
        if random.random()<self.epsilon:
            action=random.choice(self.actions)
        #Exploit
        else:
            listq=[]
            for a in self.actions:
                listq.append(self.getQ(state, a))
            
            maxq= max(listq)
            number_max= listq.count(maxq)

            if number_max>1:
                best= [i for i in range(len(self.actions)) if listq[i]==maxq]
                i= random.choice((best))
            else:
                i=listq.index(maxq)
            
            action= self.actions[i]

        if return_q:
            return action, self.q.get((state, action), 0.0)

        return action

    def learn(self, state1, action1, reward, state2):
        '''
        @brief updates the Q(state,value) dictionary using the bellman update
            equation
        '''
        # TODO: Implement the Bellman update function:
        #     Q(s1, a1) += alpha * [reward(s1,a1) + gamma* max(Q(s2)) - Q(s1,a1)]
        # 
        # NOTE: address edge cases: i.e. 
        # 
        # Find Q for current (state1, action1)
        # Address edge cases what do we want to do if the [state, action]
        #       is not in our dictionary?
        # Find max(Q) for state2
        # Update Q for (state1, action1) (use discount factor gamma for future 
        #   rewards)

        # THE NEXT LINES NEED TO BE MODIFIED TO MATCH THE REQUIREMENTS ABOVE

        max_q_new= max([self.getQ(state2,a) for a in self.actions])

        if (state1,action1) not in self.q:
            self.q[(state1,action1)]=0.0
        
        current_q=self.getQ(state1,action1)

        new_q = current_q + self.alpha*(reward + (self.gamma* max_q_new) - current_q)

        self.q[(state1,action1)] = new_q
