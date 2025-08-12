import pygame
import random as rnd
from agent import Agent, Piece
from utils import AgentAction, AgentObs, get_color, get_action_id, get_action_queue

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BACKGROUND_COLOR = (215,230,214)
GRAY = (215 - 50,230 - 50,214 - 50)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 255, 0)

# -----

def generate_piece_color(agent_color, n_colors):
    piece_color = rnd.randint(0, n_colors - 1)
    if piece_color == (agent_color - 1):
        piece_color = (piece_color + rnd.randint(1, n_colors - 1)) % n_colors
    return piece_color + 1

class GameEnv:
    def __init__(self,
                 max_steps,
                 n_colors,
                 n_agents,
                 n_pieces,
                 n_letters,
                 grid_size,
                 cell_size = 24,
                 listen_history_size = 5,
                 fps = 60):
        self.step_ = 0
        self.max_steps = max_steps
        self.fps = fps
        self.n_agents = n_agents
        self.n_letters = n_letters
        self.n_pieces = n_pieces
        self.n_colors = n_colors
        self.listen_history_size = listen_history_size

        self.grid_size = grid_size
        self.cell_size = cell_size

        self.agents : list[Agent] = []
        self.pieces : list[Piece] = []
        self.learning_agent_id = -1

        assert 0 < self.n_agents <= self.n_pieces
        assert self.n_colors > 1
        assert self.n_letters > 0

        self.ep_reward = 0

        self._init()

    def _init(self):
        pygame.init()
        pygame.font.init()

        self.screen_width, self.screen_height = self.grid_size * self.cell_size, self.grid_size * self.cell_size

        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        #self.clock = pygame.time.Clock()
        self.font_normal = pygame.font.Font('C:\Windows\Fonts\micross.ttf', 18)
        self.font_small = pygame.font.Font('C:\Windows\Fonts\micross.ttf', 14)

        print("GAME ENV INITIALIZED")

        Agent.max_agent_id = self.n_agents
        Agent.max_piece_letter = self.n_letters
        Agent.grid_size = self.grid_size
        Agent.n_colors = self.n_colors
        Agent.listen_history_size = self.listen_history_size
        Agent.vision_grid_size = 5

    def init_instances(self, model):
        self.agents.clear()
        self.learning_agent_id = rnd.randint(1, self.n_agents)
        for i in range(self.n_agents):
            if i + 1 == self.learning_agent_id:
                self.agents.append(Agent(i + 1, 0, 0, None)) # Learning agent
            else:
                self.agents.append(Agent(i + 1, 0, 0, model))

        self.reset()

    def reset(self):
        self.step_ = 0
        self.ep_reward = 0
        self.collective_reward = 0

        #TODO: generalize this for color teams of different sizes
        agent_colors = [(i % self.n_colors) + 1 for i in range(self.n_agents)]
        rnd.shuffle(agent_colors)

        self.pieces.clear()
        for i in range(self.n_pieces):
            self.pieces.append(Piece.make_new_piece(self.pieces, rnd.randint(1, self.n_letters)))
            if i >= self.n_agents:
                self.pieces[-1].color = rnd.randint(1, self.n_colors)

        for i in range(self.n_agents):
            self.pieces[i].color = generate_piece_color(agent_colors[i], self.n_colors)
            self.agents[i].reset(self.agents, agent_colors[i], self.pieces[i])

        return self.learning_agent.process_obs(self.env_info)

    def step(self, learning_agent_action : AgentAction):
        obs : AgentObs = []
        reward = 0

        obs = self.learning_agent.process_obs(self.env_info)

        if self.step_ >= self.max_steps:
            print("END OF EPISODE", "TOTAL REWARD: ", self.ep_reward + self.collective_reward)
            return obs, self.collective_reward, True, False

        # Start of Frame

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("USER QUIT !!!")
                exit() #TODO: something more elegant?
        
        # End of Input Event

        # Step Event
        env_info = self.env_info
        action_queue = {}

        action_queue = get_action_queue()

        # Choosing other agent's actions
        for agent in self.agents:
            if not agent.is_learning_agent():
                obs_ = agent.process_obs(env_info)
                action = agent.choose_action(obs_)
                action_queue[get_action_id(action)].append((agent.id, action))
            else:
                action_queue[get_action_id(learning_agent_action)].append((self.learning_agent_id, learning_agent_action))

        for action_id in action_queue:
            for pair in action_queue[action_id]:
                agent_id, action = pair
                agent = self.agents[agent_id - 1]

                if not agent.is_learning_agent():
                    self.collective_reward += agent.step(action, env_info)
                else:
                    reward = agent.step(action, env_info)

        obs = self.learning_agent.process_obs(env_info)

        # End of Step Event

        # Draw Event
        self.draw()
        # End of Draw Event

        #self.clock.tick(self.fps)
        # End of Frame
        self.step_ += 1
        self.ep_reward += reward

        return obs, reward, False, False
    
    def draw(self):
        self.screen.fill(BACKGROUND_COLOR)

        # GRID
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                color = GRAY
                pygame.draw.rect(self.screen, color,
                                (i * self.cell_size, j * self.cell_size, self.cell_size, self.cell_size),
                                width = 1)
        # AGENTS  
        for agent in self.agents:
            pygame.draw.circle(self.screen,
                                color = get_color(agent.color),
                                center = (agent.x  * self.cell_size, agent.y  * self.cell_size),
                                radius = 16)
            # piece agent wants
            pygame.draw.circle(self.screen, WHITE, ((agent.x - 1/2) * self.cell_size, (agent.y - 1/2) * self.cell_size), 10)
            piece_letter = self.font_small.render(str(agent.piece.letter), True, get_color(agent.piece.color, dark=True))
            piece_pos = (agent.x - 3/4) * self.cell_size, (agent.y - 3/4) * self.cell_size
            self.screen.blit(piece_letter, piece_pos)

        # PIECES
        for piece in self.pieces:
            piece_letter = self.font_normal.render(str(piece.letter), True, get_color(piece.color, dark = True))
            piece_pos = ((piece.x - 1/2) * self.cell_size, (piece.y - 1/2) * self.cell_size)
            self.screen.blit(piece_letter, piece_pos)

        pygame.display.flip()
    
    @property
    def env_info(self):
        return (self.agents, self.pieces)
    
    @property
    def learning_agent(self):
        return self.agents[self.learning_agent_id - 1]

