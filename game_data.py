import time

# Constants
team_black = 0
team_white = 1
keeper   = 0
defense  = 1
midfield = 2
offense  = 3

# Field class to remember the current field dimensions
class field_data:
    def __init__(self):
        self.hull = None # Defines the green area of the field
        self.center = None # Coordinates of the center of the field
        self.goals = None # Location of the two goals based on field hull
        self.regions = None # Regions around the player positions (e.g. keeper, midfield)

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

    def set_data(self, pos, speed, angle, accuracy, field_index):
        self.pos = pos
        self.speed = speed
        self.angle = angle
        self.accuracy = accuracy
        self.field_index = field_index

# Game class to remember data about the entire game
class game_data:
    # Recalculate field dimensions every N frames
    update_field_timer = 60
    def __init__(self):
        # datapoints is a list of datapoints, the last of which is always a
        # dummy that tracks the number of frames the ball has been hidden
        # As soon as the ball reappears, the dummy datapoint stores that data,
        # and a new dummy is created to take its place
        self.datapoints = [datapoint()]
        self.fields = []
        self.update_field = 0 # Recalculate field dimensions when this reaches zero
        self.score = [0, 0]
        self.time = 0 # Time at which the game started
        self.duration = 0

    # Start a game by starting the timer, This method is called automatically
    # in the track.track() function when the ball is detected for the first time
    def start(self):
        self.time = time.time()

    # Stop a game by stopping the timer and removing the final dummy datapoint,
    # This method must be called manually after the game is done
    def stop(self):
        self.duration = time.time() - self.time
        self.datapoints.pop(-1)

    def add_score(self, score, team):
        self.score[team] = max(0, self.score[team] + score)

    def curr_field(self):
        return self.fields[-1]

    # Add a new field instance to the list
    def new_field(self):
        self.fields.append(field_data())
        return self.curr_field()

    # Get the index in the field list of the current field
    def field_index(self):
        return len(self.fields) - 1

    # Add a new datapoint instance to the list
    def new_dp(self):
        self.datapoints.append(datapoint())

    # Get the most recent datapoint or none if there are no datapoints yet
    def last_seen(self):
        if self.datapoints[:-1]:
            return self.datapoints[-2]
        return None

    # Returns true if the score is such that one team should've won already
    def is_done(self):
        return abs(self.score[0] - self.score[1]) >= 2 and (self.score[0] >= 10 or self.score[1] >= 10)
