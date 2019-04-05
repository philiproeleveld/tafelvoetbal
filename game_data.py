import time

# Constants
team_black = 0
team_white = 1
keeper   = 0
defense  = 1
midfield = 2
offense  = 3

# Hit class to store data about single hits
class hit:
    def __init__(self, type, team=None, player=None):
        self.type = type
        self.team = team
        self.player = player
        self.goal = None

# Datapoint class to store data about a single frame
class datapoint:
    # __init__() creates a dummy instance (see game_data class for more information)
    # to store actual data about a datapoint use set_data() after initialization
    def __init__(self):
        self.hidden = 0
        self.hit = None

    def set_data(self, pos, speed, angle, accuracy, hull, fieldcenter):
        self.pos = pos
        self.speed = speed
        self.angle = angle
        self.accuracy = accuracy
        self.hull = hull
        self.fieldcenter = fieldcenter

# Game class to remember data about the entire game
class game_data:
    def __init__(self):
        # datapoints is a list of datapoints, the last of which is always a
        # dummy that tracks the number of frames the ball has been hidden
        # As soon as the ball reappears, the dummy datapoint stores that data,
        # and a new dummy is created to take its place
        self.datapoints = [datapoint()]
        self.score = [0, 0]
        self.time = 0 # Time at which the game started
        self.duration = 0

    def start(self):
        self.time = time.time()

    def stop(self):
        self.duration = time.time() - self.time
        self.datapoints.pop(-1)

    def add_score(self, score, team):
        self.score[team] = max(0, self.score[team] + score)

    def add_dp(self):
        self.datapoints.append(datapoint())

    def last_seen(self):
        if self.datapoints[:-1]:
            return self.datapoints[-2]
        return None

    def is_done(self):
        return abs(self.score[0] - self.score[1]) >= 2 and (self.score[0] >= 10 or self.score[1] >= 10)
