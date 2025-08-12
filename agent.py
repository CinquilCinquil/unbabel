from enum import Enum
from utils import (AgentAction, AgentObs,
    calculate_dis, get_random_pos, get_emtpy_speech, cell_free, unpack_action, drop_piece_in_room)

reward_config = {
    "invalid_movement_penalty" : 1,
    "invalid_pick_up_penalty" : 1,
    "invalid_offer_penalty" : 1,
    "invalid_stop_offering_penalty" : 1,
    "invalid_drop_piece_penalty" : 1,
    "successful_offer_reward" : 1,
    "accept_piece_reward" : 3,
    "piece_found_reward" : 20,
    "teammate_piece_found_reward" : 0.5,
}

class AgentActions(Enum):
    MOVE = 0
    PICK_UP_A_PIECE = 1
    OFFER_A_PIECE = 2
    ACCEPT_A_PIECE = 3
    STOP_OFFERING_A_PIECE = 4
    DROP_PIECE = 5
    SPEAK = 6

class Agent:

    max_agent_id = 0
    max_piece_letter = 0
    grid_size = 0
    n_colors = 0
    listen_history_size = 0
    vision_grid_size = 0

    def __init__(self, id, x, y, model):
        self.id = id
        self.x = x
        self.y = y
        self.color = None
        self.piece = None
        self.piece_in_hand = None
        self.piece_being_offered = None
        self.agent_with_offer = None
        
        self.model = model
        self.reward = 0
        self.my_speech = get_emtpy_speech() # what i said this turn
        self.listen_history = [get_emtpy_speech() for i in range(Agent.listen_history_size)]

    def step(self, action : AgentAction, env_info) -> float:
        discrete_action, dx, dy, agent_ind, speech = unpack_action(action)
        agents, pieces = env_info
        self._reset_speech()

        if AgentActions.MOVE.value == discrete_action:
            self.move(dx, dy, agents)
        elif AgentActions.PICK_UP_A_PIECE.value == discrete_action:
            self.pick_up_a_piece(pieces)
        elif AgentActions.OFFER_A_PIECE.value == discrete_action:
            self.offer_a_piece(agents, agent_ind)
        elif AgentActions.ACCEPT_A_PIECE.value == discrete_action:
            self.accept_a_piece(agents, pieces, agent_ind)
        elif AgentActions.STOP_OFFERING_A_PIECE.value == discrete_action:
            self.stop_offering_a_piece(agents, agent_ind)
        elif AgentActions.DROP_PIECE.value == discrete_action:
            self.drop_piece(agents, pieces)
        elif AgentActions.SPEAK.value == discrete_action:
            self.speak(speech)
        else:
            ... # do nothing

        if self.piece_in_hand != None:
            self.piece_in_hand.x = self.x
            self.piece_in_hand.y = self.y

        return self._get_reward()

    def process_obs(self, env_info) -> AgentObs:
        agents, pieces = env_info

        self._process_listen_history(agents)
        vision_grid = self._process_vision_grid(agents, pieces)
        offer = self._process_offer(agents)
        
        obs_dict = {"eyes" : vision_grid, "offer" : offer, "desired_piece" : self.desired_piece}
        for i in range(Agent.listen_history_size):
            obs_dict["speech " + str(i + 1)] = self.listen_history[i]

        return obs_dict

    def choose_action(self, obs : AgentObs) -> AgentAction:
        action, _ = self.model.predict(obs)
        return action

    def is_learning_agent(self):
        return self.model == None

    def move(self, dx : int, dy : int, agents : list) -> None:
        new_x = self.x + dx
        new_y = self.y + dy

        in_bounds = new_x < Agent.grid_size and new_y < Agent.grid_size and 0 <= new_x and 0 <= new_y

        if in_bounds and cell_free(new_x, new_y, agents):
            self.x, self.y = new_x, new_y
        elif self.is_learning_agent():
            self.reward -= reward_config["invalid_movement_penalty"]

    def pick_up_a_piece(self, pieces):
        if self.piece_in_hand == None:
            for piece in pieces:
                if piece.x == self.x and piece.y == self.y:
                    if piece.color == self.color:
                        self.piece_in_hand = piece
                    elif self.is_learning_agent():
                        self.reward -= reward_config["invalid_pick_up_penalty"]
                    break

    def offer_a_piece(self, agents, agent_ind):
        if agent_ind != self.id - 1 and self.piece_in_hand != None:
            target_agent = agents[agent_ind]

            target_not_being_offered = target_agent.agent_with_offer == None
            target_in_distance = calculate_dis(self.x, self.y, target_agent.x, target_agent.y) <= Agent.vision_dis()

            reward = 0
            if target_not_being_offered and target_in_distance:
                target_agent.agent_with_offer = self.id
                target_agent.piece_being_offered = self.piece_in_hand
                reward += reward_config["successful_offer_reward"]
            else:
                reward -= reward_config["invalid_offer_penalty"]

            if self.is_learning_agent():
                self.reward += reward

    def accept_a_piece(self, agents, pieces, agent_ind):
        target_agent = agents[agent_ind]
        if self.piece_being_offered != None and target_agent.piece_in_hand == self.piece_being_offered:

            if self.is_learning_agent():
                self.reward += reward_config["accept_piece_reward"]
            
            if self.piece_in_hand != None:
                self.drop_piece(agents, pieces)
            
            if (self.piece_being_offered.color == self.piece.color
                and self.piece_being_offered.letter == self.piece.letter):

                if self.is_learning_agent():
                    self.reward += reward_config["piece_found_reward"]
                else:
                    self.reward += reward_config["teammate_piece_found_reward"]
                
                pieces.remove(self.piece_being_offered)
                print("Piece found!")
            else:
                print("Trade!")
                self.piece_in_hand = self.piece_being_offered
            
            self.piece_being_offered = None
            self.agent_with_offer = None
            target_agent.piece_in_hand = None
    
    def stop_offering_a_piece(self, agents, agent_ind):
        target_agent = agents[agent_ind]
        if self.piece_in_hand != None and target_agent.piece_being_offered == self.piece_in_hand:
            self.agent_with_offer = None
            target_agent.piece_being_offered = None
        elif self.is_learning_agent():
            self.reward -= reward_config["invalid_stop_offering_penalty"]

    def drop_piece(self, agents, pieces):
        if self.piece_in_hand != None:
            drop_piece_in_room(self.piece_in_hand, agents + pieces, Agent.grid_size)
            self.piece_in_hand = None
        elif self.is_learning_agent():
            self.reward -= reward_config["invalid_drop_piece_penalty"]

    def speak(self, speech):
        self.my_speech = speech
    
    def reset(self, agents, color, piece):
        for i in range(Agent.listen_history_size):
            self.listen_history[i] = get_emtpy_speech()
        self._reset_speech()
        self.reward = 0

        self.x, self.y = get_random_pos(agents, Agent.grid_size)
        self.color = color
        self.piece = piece
        self.piece_in_hand = None
        self.piece_being_offered = None
        self.agent_with_offer = None

    def _reset_speech(self):
        self.my_speech = get_emtpy_speech()

    def _get_reward(self):
        reward = self.reward 
        self.reward = 0
        return reward

    def _process_vision_grid(self, agents, pieces):
        # (agent id, agent color, piece letter, piece color,
        # letter of piece that agent is holding, color of piece that agent is holding)
        vision_grid = [[(0, 0, 0, 0, 0, 0) for i in range(Agent.vision_grid_size)]
                        for j in range(Agent.vision_grid_size)]

        for agent in agents:
            if calculate_dis(agent.x, agent.y, self.x, self.y) <= Agent.vision_dis():
                x_, y_ = agent.x - (self.x - 2), agent.y - (self.y - 2)
                _, _, piece_letter, piece_color, _, _  = vision_grid[x_][y_]

                vision_grid[x_][y_] = (
                    agent.id / Agent.max_agent_id,
                    agent.color / Agent.n_colors,
                    piece_letter,
                    piece_color,
                    0 if agent.piece_in_hand == None else agent.piece_in_hand.letter / Agent.max_piece_letter,
                    0 if agent.piece_in_hand == None else agent.piece_in_hand.color / Agent.n_colors)

        for piece in pieces:
            if calculate_dis(piece.x, piece.y, self.x, self.y) <= Agent.vision_dis():
                x_, y_ = piece.x - (self.x - 2), piece.y - (self.y - 2)
                agent_id, agent_color, _, _, piece_in_hand_letter, piece_in_hand_color = vision_grid[x_][y_]

                vision_grid[x_][y_] = (agent_id,
                    agent_color,
                    piece.letter / Agent.max_piece_letter,
                    piece.color / Agent.n_colors,
                    piece_in_hand_letter,
                    piece_in_hand_color)
                
        return vision_grid

    def _process_listen_history(self, agents):
        speech = None
        min_dis = 10000
        for agent in agents:
            if agent.id != self.id and not agent.my_speech.any():
                dis_to_agent = calculate_dis(agent.x, agent.y, self.x, self.y)
                if dis_to_agent < min_dis <= 5:
                    min_dis = dis_to_agent
                    speech = agent.my_speech

        if speech != None:
            self.listen_history.append(speech)
        
        if len(self.listen_history) > Agent.listen_history_size:
            self.listen_history.pop(0)

    def _process_offer(self, agents):
        # piece letter, piece color, target agent, am i holding a piece
        offer = [(
            self.piece_being_offered.letter / Agent.max_piece_letter if self.piece_being_offered != None else 0,
            self.piece_being_offered.color / Agent.n_colors if self.piece_being_offered != None else 0,
            self.agent_with_offer / Agent.max_agent_id if self.agent_with_offer != None else 0,
            float(self.piece_in_hand != None)
        )]

        return offer

    @property
    def desired_piece(self):
        return [(
            self.piece.letter / Agent.max_piece_letter,
            self.piece.color / Agent.n_colors
        )]

    def vision_dis():
        return int(Agent.vision_grid_size/2)

class Piece:
    def __init__(self, x, y, color, letter):
        self.x = x
        self.y = y
        self.color = color
        self.letter = letter

    def make_new_piece(pieces, letter):
        x, y = get_random_pos(pieces, Agent.grid_size)
        return Piece(x, y, None, letter) # Set color later
