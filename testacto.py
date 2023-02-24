from future.moves import itertools

dic_map_action = {}

# action_list = [p for p in itertools.product([0, 1, 2, 3, 4, 5], repeat=3)]
action_list = [p for p in itertools.product([0, 2, 3], repeat=3)]
for action in range(len(action_list)):
    dic_map_action[action] = action_list[action]
df = open('plots/action_mapp.txt', 'w')
df.write(str(dic_map_action))