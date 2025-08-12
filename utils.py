import numpy as np
import random as rnd
from typing import Dict, Union

AgentAction = Dict[int, Union[int, np.ndarray]]
AgentObs = Dict[int, np.ndarray]

def unpack_action(action : AgentAction):
    if type(action) is not dict:
        action = dict(action)
    
    discrete_action = action["action"]
    dx, dy = action["dx"] - 1, action["dy"] - 1
    agent_ind = action["agent"]
    speech = action["speech"]
    return discrete_action, dx, dy, agent_ind, speech

def get_action_id(action : AgentAction):
    discrete_action, _, _, _, _ = unpack_action(action)
    return discrete_action

#TODO: this should rely on action_space info
def get_action_queue():
    action_queue = {}
    for i in range(8):
        action_queue[i] = []
    return action_queue

def calculate_dis(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def get_random_pos(instances, grid_size):
    for i in range(grid_size ** 2):
        x, y = rnd.randint(0, grid_size - 1), rnd.randint(0, grid_size - 1)
        return_ = True
        for instance in instances:
            if instance.x == x and instance.y == y:
                return_ = False
                break
        if return_:
            return x, y
    raise Exception("EMPTY CELL NOT FOUND")

#TODO: replace action and obs space definitions in code
def get_emtpy_speech():
    return np.array([[0 for i in range(120)]], dtype=np.float32)

def cell_free(x, y, agents):
    for agent in agents:
        if agent.x == x and agent.y == y:
            return False
    return True

def drop_piece_in_room(piece, instances, grid_size, ):
    new_x, new_y = get_random_pos(instances, grid_size)
    piece.x = new_x
    piece.y = new_y

color_dict = {
    0 : (255, 255, 255),
    1 : (255, 0, 0),
    2 : (0, 255, 0),
    3 : (0, 0, 255),
}

def get_color(color, dark = False):
    if dark:
        r, g, b  = color_dict[color % (1 + len(color_dict))]
        return (r//2, g//2, b//2)
    else:
        return color_dict[color % (1 + len(color_dict))]