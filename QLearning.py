import random
import numpy as np

# drawing library
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# set up environment
# initiate graph with grid line
fig = plt.figure()
ax = fig.add_subplot(111)
ax.autoscale(False)

# x_axis limit (-15,15), y_axis limit (-15,15)
# Major ticks every 5, minor ticks every 1
major_ticks = np.arange(0, 12, 2)
minor_ticks = np.arange(0, 12, 2)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)

# Turn off tick labels
ax.set_yticklabels([])
ax.set_xticklabels([])

# grid setting
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.5)

# class responsible for creating environmental object
class environment():
    def __init__(self, state_coor):
        self.stateCoor = state_coor
        self.environmentCoor = []

    def getstateCoor(self):
        return self.stateCoor

    def putEnviromentCoor(self, envCoor):
        self.environmentCoor = envCoor

    def getEnviromentCoor(self):
        return self.environmentCoor

# define state base for environment
envMap = []
m = 4
n = 4
rec_low_x = 8
rec_low_y = 8
rec_up_x = 10
rec_up_y = 10
while m >= 0:
    while n >= 0:
        entry = environment([m,n])
        entry.putEnviromentCoor([rec_low_x, rec_low_y, rec_up_x, rec_up_y])
        envMap.append(entry)
        n  -= 1
        rec_low_x -= 2
        rec_up_x -= 2
    n = 4
    m -= 1
    rec_low_x = 8
    rec_up_x = 10
    rec_up_y -= 2
    rec_low_y -= 2

# this class responsible for creating state object
class state():
    def __init__(self,coordinate):
        self.coordination = coordinate
        self.actionList = []
        self.reward = []
        self.bestAction = []
        self.q_value = []

    # input list of action for state object
    def putAction(self,action):
        self.actionList = action

    # input reward for state object
    def putReward(self,stateReward):
        self.reward = stateReward

    # input best action for state object
    def putBestAction(self,best):
        self.bestAction = best

    # remove an action from action list
    def removeAction(self,action):
        for i in self.actionList:
            if i == action:
                self.actionList.remove(i)

    # input/update action with q value for state object
    # format q_value = [[action1,qValue1], [action2,qValue2], ...]
    def putQvalue(self,actionQValuePair):
        # this flag check if the actionQValuePair is new (aka has not been registered in q_value list)
        check_new = True
        # check if qvalue list is empty
        if self.q_value:
            # if not empty and the qvalue already associated with an action then update its qvalue with new qvalue
            for i in self.q_value:
                if i[0] == actionQValuePair[0]:
                    i[1] = actionQValuePair[1]
                    check_new = False

            # if not empty and qvalue has not associated with an action then register both action and new qvalue
            if check_new == True:
                self.q_value.append(actionQValuePair)

        elif not self.q_value:
            self.q_value.append(actionQValuePair)


    # return q value given an action of the state object
    def getQvalue(self,action):
        for i in self.q_value:
            if i[0] == action:
                return i[1]

    # return coordinate of state
    def getCoor(self):
        return self.coordination

    # return a list of available action of state object
    def getActionList(self):
        return self.actionList

    # return reward of state object
    def getReward(self):
        return self.reward

    # return best action of state object
    def getBestAction(self):
        # initiate q value
        max_qValue = -100
        for i in self.q_value:
            if i[1] >= max_qValue:
                max_qValue = i[1]
                self.bestAction = i[0]
        return self.bestAction

    # return random action of state object
    def getRandomAction(self):
        check_index = False
        n = random.randint(0,4)
        while check_index == False:
            if n < len(self.actionList):
                check_index = True
                return self.actionList[n]
            else:
                n = random.randint(0,4)

# function responsible for process next state given a current state and action
def moveState(currentState,action):
    if action == 'U':
        x = currentState[0] + 1
        y = currentState[1]
    elif action == 'D':
        x = currentState[0] - 1
        y = currentState[1]
    elif action == 'L':
        x = currentState[0]
        y = currentState[1] - 1
    elif action == 'R':
        x = currentState[0]
        y = currentState[1] + 1
    elif action == 'S':
        x = currentState[0]
        y = currentState[1]
    nextState = [x,y]
    return nextState

# function responsible for getting information of a state
def getStateInfo(state, stateList):
    # this flag check if we have reached the goal state
    goal = False
    for i in stateList:
        if i.getCoor() == state:
            coordinate =i.getCoor() # Coordinate of the state
            reward = i.getReward() # Reward of the state
            best_q_value = i.getQvalue(i.getBestAction()) # Q value of the state with the current best action
            if i.getCoor() == [2,2]:
                goal = True
            return coordinate, reward, best_q_value, goal
    return "Null", "Null", "Null"

# define value
min_row = 0
max_row = 4
min_col = 0
max_col = 4

total_episodes = 100          # Total episodes
episodes_list = []            # List of episodes
total_test_episodes = 20      # Total test episodes
test_episodes_list = []       # List of test episodes
max_steps = 100               # Max steps per episode

learning_rate = 0.7           # Learning rate
gamma = 0.618                 # Discounting rate

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability
decay_rate = 0.01             # Exponential decay rate for exploration prob

# initiate qtable
qtable =[]

# create a 5x5 states table
for m in range (0,5):
    for n in range (0,5):
        newState = state([m,n])
        qtable.append(newState)


# assigned action for each state depend on its position
# possible actions are U: Up, D: Down, L: Left, R: Right, S: Stay
for i in qtable:
        i.putAction(['U','D','L','R','S'])
        if i.getCoor()[0] == min_row:
            i.removeAction('D')
        if i.getCoor()[0] == max_row:
            i.removeAction('U')
        if i.getCoor()[1] == min_col:
            i.removeAction('L')
        if i.getCoor()[1] == max_col:
            i.removeAction('R')

'''
# debug#######################
for i in qtable:
    print(i.getCoor())
    print(i.getActionList())
##############################
'''


# assigned reward for each state depend on its position
for i in qtable:
    if i.getCoor() == [2,2]:
        i.putReward(5)
    else:
        i.putReward(-1)

'''
# debug#######################
for i in qtable:
    print(i.getCoor())
    print(i.getActionList())
    print(i.getReward())
##############################
'''

# initiate q value = 0 for all possible action  for each state
for i in qtable:
    for m in i.getActionList():
        i.putQvalue([m,0])

# initiate best action for each state, initially = first available action in state action list
#for i in qtable:
#    i.putBestAction(i.actionList[0])

'''
# debug#######################
for i in qtable:
    #print(i.getCoor())
    #print(i.getActionList())
    #print(i.getBestAction())
    #print(i.getRandomAction())
    if 'U' in i.getActionList():
        print(i.getCoor())
        print(i.getActionList())
        print(i.getQvalue('U'))
##############################
'''

# create episode (aka difference situation the robot start
# train episode:
for i in range (total_episodes):
    m = random.randint(0,4)
    n = random.randint(0,4)
    starting_state = [m,n]
    episodes_list.append(starting_state)

# test episode:
for i in range (total_test_episodes):
    m = random.randint(0,4)
    n = random.randint(0,4)
    starting_state = [m,n]
    test_episodes_list.append(starting_state)

'''
# debug#######################
coor,reward,qValue,goal = getStateInfo([1,2],qtable)
print(coor)
print(reward)
print(qValue)
##############################
'''


# start training
for episode in range (total_episodes):

    # starting state from the episode scenario
    start_state = episodes_list[episode]
    # current state
    current_state = start_state

    # a series of action step in an episode
    for step in range (max_steps):
        # this flag check if we have yet to reached the goal state
        goal = False

        # iterate through q table
        for i in qtable:
            if i.getCoor() == current_state:

                # generate a random number between (0,1)
                decider = random.uniform(0,1)

                # if this number greater than exploration parameter epsilon, we will do exploitation (taking the action with biggest Q value for this state)
                if decider > epsilon:
                    action = i.getBestAction()
                # else choose a random action (aka exploration)
                else:
                    action = i.getRandomAction()

                # initiate next state parameters
                nextState_coordinate = []
                nextState_reward = []
                nextState_q_value = []
                current_q_value = []


                # move to the next state given action and current state
                nextState = moveState(current_state, action)
                for getNextState in qtable:
                    if getNextState.getCoor() == nextState:
                        nextState_coordinate, nextState_reward, nextState_best_q_value, goal = getStateInfo(nextState, qtable)

                # calculate Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
                current_q_value = i.getQvalue(action) + learning_rate * (nextState_reward + gamma * nextState_best_q_value - i.getQvalue(action))
                # update Q(s,a)
                i.putQvalue([action,current_q_value])

                # Our new state is now state
                current_state = nextState

        # check if we have reach the goal state:
        if goal == True:
            break

    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)


# debug#######################
for i in qtable:
    print(i.getCoor())
    print(i.getActionList())
    print(i.getBestAction())
    print(i.getQvalue(i.getBestAction()))
##############################



# testing the agent
total_reward = []
output_file = open("Testlog.txt", "w+")
for episode in range(total_test_episodes):
    current_test_state = test_episodes_list[episode]
    test_goal = False
    best_action_series = []
    episode_reward = 0
    # print output to txt file
    output_file.write("****************************************************\r\n")
    print("****************************************************")
    output_file.write("EPISODE %s\r\n" % episode)
    print("EPISODE",episode)
    output_file.write("Starting position: %s\r\n" % current_test_state )
    print("Starting position: ", current_test_state)


    for step in range (max_steps):
        for i in qtable:
            # get the agent start at the test state
            if i.getCoor() == current_test_state:
                for j in envMap:
                    if j.getstateCoor() == current_test_state:
                        m = (j.getEnviromentCoor()[2] + j.getEnviromentCoor()[0])/2
                        n = (j.getEnviromentCoor()[3] + j.getEnviromentCoor()[1])/2
                        if current_test_state == test_episodes_list[episode]:
                            point = ax.scatter(m, n, color='red', marker='o', zorder=6)
                            plt.pause(1)
                        else:
                            point = ax.scatter(m, n, color='blue', marker='o', zorder=6)
                            plt.pause(0.5)

                # choose the best action to move next
                action = i.getBestAction()
                best_action_series.append(action)

                # move to the next state given action and current state
                advance_next_state = moveState(current_test_state, action)
                for getNextState in qtable:
                    if getNextState.getCoor() == advance_next_state:
                        next_test_state_coordinate, next_test_state_reward, next_test_state_best_q_value, test_goal = getStateInfo(advance_next_state, qtable)

                current_test_state = advance_next_state

                episode_reward += next_test_state_reward


                point.remove()

                if test_goal == True:
                    # draw the goal point
                    for j in envMap:
                        if j.getstateCoor() == next_test_state_coordinate:
                            m = (j.getEnviromentCoor()[2] + j.getEnviromentCoor()[0]) / 2
                            n = (j.getEnviromentCoor()[3] + j.getEnviromentCoor()[1]) / 2
                            point = ax.scatter(m, n, color='green', marker='o', zorder=6)
                            plt.pause(1)
                            point.remove()
                    break


        if test_goal == True:
            total_reward.append(episode_reward)
            output_file.write("List of actions: %s\r\n" % best_action_series)
            print("List of actions: ",best_action_series)
            output_file.write("Episode rewards: %d\r\n" % episode_reward)
            print("Episode rewards: ",episode_reward)
            break

output_file.write ("****************************************************\r\n")
output_file.write ("Score over time: %s\r\n" % str(sum(total_reward)/total_test_episodes))
print("Score over time: ", str(sum(total_reward)/total_test_episodes))
# draw the graph
plt.show()
output_file.close()







































