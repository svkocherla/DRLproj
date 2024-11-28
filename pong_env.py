import sys
import random
import pygame
import numpy as np
from pygame.locals import *
import gym
from gym import error, spaces, utils
# from gym.utils import seeding

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


class PongSinglePlayerEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, ball_speed=4, bat_speed=4, max_num_rounds=20):
        SCREEN_WIDTH, SCREEN_HEIGHT = 160, 210

        # ballx, bally, left, right, speedx, speedy
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, -ball_speed, -ball_speed]),
            high=np.array([max(SCREEN_HEIGHT, SCREEN_WIDTH), max(SCREEN_HEIGHT, SCREEN_WIDTH), 
                        max(SCREEN_HEIGHT, SCREEN_WIDTH), max(SCREEN_HEIGHT, SCREEN_WIDTH), 
                        ball_speed, ball_speed]),
            shape=(6,)
        )
        
        self.action_space = spaces.Discrete(3)

        pygame.init()
        self._surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        self._viewer = None
        self._game = PongGame(
            has_double_players=False,
            window_size=(SCREEN_WIDTH, SCREEN_HEIGHT),
            ball_speed=ball_speed,
            bat_speed=bat_speed,
            max_num_rounds=max_num_rounds)

    def step(self, action):
        assert self.action_space.contains(action)
        bat_directions = [-1, 0, 1]
        rewards, done = self._game.step(bat_directions[action], None)
        obs = self._get_state()
        return (obs, rewards[0], done, {})

    def reset(self):
        self._game.reset_game()
        obs = self._get_state()
        return obs
    
    def _get_state(self):
        ballx, bally = self._game._ball._rect.x, self._game._ball._rect.y
        speedx, speedy = self._game._ball.speed_x, self._game._ball.speed_y
        left = self._game._left_bat._rect.y
        right = self._game._right_bat._rect.y
        return np.array([ballx, bally, left, right, speedx, speedy])
        

class PongDoublePlayerEnv(PongSinglePlayerEnv):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, ball_speed=4, bat_speed=4, max_num_rounds=20):
        SCREEN_WIDTH, SCREEN_HEIGHT = 160, 210
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, -ball_speed, -ball_speed]),
            high=np.array([max(SCREEN_HEIGHT, SCREEN_WIDTH), max(SCREEN_HEIGHT, SCREEN_WIDTH), 
                        max(SCREEN_HEIGHT, SCREEN_WIDTH), max(SCREEN_HEIGHT, SCREEN_WIDTH), 
                        ball_speed, ball_speed]),
            shape=(6,)
        )
        self.action_space = spaces.Discrete(3)
        self.double_action_space = spaces.Tuple(
            [spaces.Discrete(3), spaces.Discrete(3)])

        pygame.init()
        self._surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

        self._viewer = None
        self._game = PongGame(
            has_double_players=True,
            window_size=(SCREEN_WIDTH, SCREEN_HEIGHT),
            ball_speed=ball_speed,
            bat_speed=bat_speed,
            max_num_rounds=max_num_rounds)

    def step(self, action):
        assert self.double_action_space.contains(action)
        left_player_action, right_player_action = action
        bat_directions = [-1, 0, 1]
        rewards, done = self._game.step(bat_directions[left_player_action],
                                        bat_directions[right_player_action])
        obs = self._get_state()
        return (obs, rewards, done, {})

    def reset(self):
        self._game.reset_game()
        obs = self._get_state()
        return obs
    
    def _get_state(self):
        ballx, bally = self._game._ball._rect.x, self._game._ball._rect.y
        speedx, speedy = self._game._ball.speed_x, self._game._ball.speed_y
        left = self._game._left_bat._rect.y
        right = self._game._right_bat._rect.y
        return np.array([ballx, bally, left, right, speedx, speedy])


class PongGame():
    def __init__(self,
                 has_double_players=False,
                 window_size=(160, 210),
                 top_border_thickness=34,
                 bat_height=15,
                 bat_width=5,
                 bat_offset=16,
                 ball_size=4,
                 ball_speed=1,
                 bat_speed=1,
                 max_num_rounds=20,
                 max_step_per_round=1000):
        self._max_num_rounds = max_num_rounds
        self._has_double_players = has_double_players
        self._max_step_per_round = max_step_per_round
        self._num_rounds = 0

        self._arena = Arena(window_size, top_border_thickness)
        self._ball = Ball(self._arena.centerx - ball_size // 2,
                          self._arena.centery - ball_size // 2, ball_size,
                          ball_speed)
        self._left_bat = Bat(bat_offset, self._arena.centery - bat_height // 2,
                             bat_width, bat_height, bat_speed)
        if self._has_double_players:
            self._right_bat = Bat(self._arena.right - bat_width - bat_offset,
                                  self._arena.centery - bat_height // 2,
                                  bat_width, bat_height, bat_speed)
        else:
            self._right_bat = AutoBat(
                self._arena.right - bat_width - bat_offset,
                self._arena.centery - bat_height // 2, bat_width, bat_height,
                bat_speed)
        self._scoreboard = Scoreboard(20, 8, font_size=20)

        self.reset_game()

    def step(self, left_bat_move_dir, right_bat_move_dir):
        self._num_steps += 1
        self._left_bat.move(self._arena, left_bat_move_dir)
        if self._has_double_players:
            self._right_bat.move(self._arena, right_bat_move_dir)
        else:
            self._right_bat.move(self._arena, self._ball)
        self._ball.move(self._arena, self._left_bat, self._right_bat)

        if self._ball.left_out_of_arena(self._arena):
            self._score_right += 1
            rewards = (-1, 1)
            self._reset_round()
        elif self._ball.right_out_of_arena(self._arena):
            self._score_left += 1
            rewards = (1, -1)
            self._reset_round()
        elif self._num_steps > self._max_step_per_round:
            print("Time out to be a tie round.")
            rewards = (0, 0)
            self._reset_round()
        else:
            rewards = (0, 0)

        if self._num_rounds >= self._max_num_rounds:
            done = True
        else:
            done = False
        return rewards, done

    def _reset_round(self):
        self._ball.reset()
        self._num_rounds += 1
        self._num_steps = 0

    def reset_game(self):
        self._score_left, self._score_right = 0, 0
        self._reset_round()
        self._left_bat.reset()
        self._right_bat.reset()
        self._num_rounds = 0

    def draw(self, surface):
        self._arena.draw(surface)
        self._ball.draw(surface)
        self._left_bat.draw(surface)
        self._right_bat.draw(surface)

    def draw_scoreboard(self, surface):
        self._scoreboard.draw(surface, self._score_left, self._score_right)


class Arena(pygame.sprite.Sprite):
    def __init__(self, window_size, top_border_thickness):
        window_width, window_height = window_size
        self._rect = pygame.Rect(0, top_border_thickness, window_width,
                                 window_width)

    def draw(self, surface):
        surface.fill(WHITE)
        pygame.draw.rect(surface, BLACK, self._rect)

    @property
    def left(self):
        return self._rect.left

    @property
    def right(self):
        return self._rect.right

    @property
    def top(self):
        return self._rect.top

    @property
    def bottom(self):
        return self._rect.bottom

    @property
    def centerx(self):
        return self._rect.centerx

    @property
    def centery(self):
        return self._rect.centery

class Ball(pygame.sprite.Sprite):
    def __init__(self, x, y, size, speed):
        self._x_init, self._y_init = x, y
        self._speed = speed
        self._rect = pygame.Rect(x, y, size, size)
        self.reset()

    def reset(self):
        self._rect.x = self._x_init
        self._rect.y = self._y_init
        init_speed_x = float(self._speed)
        init_speed_y = random.uniform(self._speed * 0.3, self._speed)
        self._speed_x = random.choice([-init_speed_x, init_speed_x])
        self._speed_y = random.choice([-init_speed_y, init_speed_y])

    def draw(self, surface):
        pygame.draw.rect(surface, WHITE, self._rect)

    def move(self, arena, left_bat, right_bat):
        prev_ball_left = self._rect.left
        prev_ball_right = self._rect.right
        y_on_right_bat = (right_bat.left - self._rect.right) / self._speed_x * \
                self._speed_y + self._rect.y
        y_on_left_bat = (left_bat.right - self._rect.left) / self._speed_x * \
                self._speed_y + self._rect.y

        self._rect.x += self._speed_x
        self._rect.y += self._speed_y

        if self._speed_y < 0 and self._rect.top <= arena.top:
            self._bounce('y', 0)
            self._rect.top = arena.top
        elif self._speed_y > 0 and self._rect.bottom >= arena.bottom:
            self._bounce('y', 0)
            self._rect.bottom = arena.bottom
        elif (self._speed_x < 0 and self._rect.left <= left_bat.right and
              y_on_left_bat + self._rect.height >= left_bat.top and
              y_on_left_bat <= left_bat.bottom and
              prev_ball_left > left_bat.right):
            self._bounce('x', left_bat.current_move * 0.7)
            self._rect.left = left_bat.right
            self._rect.y = y_on_left_bat
        elif (self._speed_x > 0 and self._rect.right >= right_bat.left and
              y_on_right_bat + self._rect.height >= right_bat.top and
              y_on_right_bat <= right_bat.bottom and
              prev_ball_right < right_bat.left):
            self._bounce('x', right_bat.current_move * 0.7)
            self._rect.right = right_bat.left
            self._rect.y = y_on_right_bat

    def left_out_of_arena(self, arena):
        if self._rect.left < arena.left:
            return True
        else:
            return False

    def right_out_of_arena(self, arena):
        if self._rect.right > arena.right:
            return True
        else:
            return False

    def _bounce(self, axis, speed_delta):
        if axis == 'x':
            self._speed_x *= -1
            self._speed_y += speed_delta
        elif axis == 'y':
            self._speed_y *= -1
            self._speed_x += speed_delta

    @property
    def speed_x(self):
        return self._speed_x

    @property
    def speed_y(self):
        return self._speed_y

    @property
    def centerx(self):
        return self._rect.centery

    @property
    def centery(self):
        return self._rect.centery


class Bat(pygame.sprite.Sprite):
    def __init__(self, x, y, width, height, speed):
        self._x_init, self._y_init = x, y
        self._speed = speed
        self._current_move = 0
        self._rect = pygame.Rect(x, y, width, height)

    def draw(self, surface):
        pygame.draw.rect(surface, WHITE, self._rect)

    def move(self, arena, direction):
        self._current_move = direction * self._speed
        self._rect.y += self._current_move
        if self._rect.bottom > arena.bottom:
            self._rect.y += arena.bottom - self._rect.bottom
        elif self._rect.top < arena.top:
            self._rect.y += arena.top - self._rect.top

    def reset(self):
        self._rect.x = self._x_init
        self._rect.y = self._y_init

    @property
    def left(self):
        return self._rect.left

    @property
    def right(self):
        return self._rect.right

    @property
    def top(self):
        return self._rect.top

    @property
    def bottom(self):
        return self._rect.bottom

    @property
    def current_move(self):
        return self._current_move


class AutoBat(Bat):
    def move(self, arena, ball):
        #If ball is moving away from paddle, center bat
        if ball.speed_x < 0:
            if self._rect.centery < arena.centery:
                self._rect.y += self._speed
            elif self._rect.centery > arena.centery:
                self._rect.y -= self._speed
        #if ball moving towards bat, track its movement.
        elif ball.speed_x > 0:
            if self._rect.centery < ball.centery:
                self._rect.y += self._speed
            else:
                self._rect.y -= self._speed
        if self._rect.bottom > arena.bottom:
            self._rect.y += arena.bottom - self._rect.bottom
        elif self._rect.top < arena.top:
            self._rect.y += arena.top - self._rect.top


class Scoreboard():
    def __init__(self, x, y, font_size=20):
        self._x = x
        self._y = y
        self._font = pygame.font.Font('freesansbold.ttf', font_size)

    def draw(self, surface, score_left, score_right):
        result_surf = self._font.render('Score = %d : %d' %
                                        (score_left, score_right), True, BLACK)
        rect = result_surf.get_rect()
        rect.topleft = (self._x, self._y)
        surface.blit(result_surf, rect)


def main():
    pygame.init()
    pygame.display.set_caption('Pong')
    pygame.mouse.set_visible(0)  # make cursor invisible
    surface = pygame.display.set_mode((160, 210))
    fps_clock = pygame.time.Clock()

    game = PongGame(window_size=(160, 210), has_double_players=True)
    ball_positions_x = []

    while True:  # main game loop
        for event in pygame.event.get():
            if event.type == QUIT:
                print(f"Min ball x: {min(ball_positions_x)}")
                print(f"Max ball x: {max(ball_positions_x)}")
                pygame.quit()
                sys.exit()

        left_bat_action = random.choice([-1, 0, 1])
        right_bat_action = random.choice([-1, 0, 1])
        _, done = game.step(left_bat_action, right_bat_action)
        ball_positions_x.append(game._ball._rect.x)

        if done:
            game.reset_game()

        game.draw(surface)
        pygame.display.update()
        fps_clock.tick(60)



if __name__ == '__main__':
    main()
