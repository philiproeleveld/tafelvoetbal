import time
import json
import MySQLdb
from datetime import timedelta

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

    # Make hull coordinates relative to the field center at that time
    def relativize_hull(self):
        self.hull = [(coord[0] - self.center[0], coord[1] - self.center[1]) for coord in self.hull]

    # Returns a JSON formatted string
    def hull_to_string(self):
        return json.dumps(self.hull.tolist())

# Hit class to store data about single hits
class hit:
    def __init__(self, type, team=None, player=None, goal=None):
        self.type = type # of type (speed_hit, angle_hit) both booleans
        self.team = team
        self.player = player
        self.goal = goal
        # self.goal is equal to either team_white or team_black if not None
        # meaning which goal the ball entered, *not* which team gains points

    # The values of the object are encoded as an integer as follows:
    # type -> 00/01/10/11;
    # encoding (False, False)/(True, False)/(False, True)/(True, True)
    # team -> 00/01/10; encoding black/white/None
    # player -> 00/01/10/11;
    # encoding encoding keeper/defense/midfield/offense
    # if team is None, so is player (i.e. ignore the player bits)
    # goal -> 00/01/10; encoding black goal/white goal/neither
    # the resulting bits are then concatenated to create an integer between
    # 0 and 183 (incl.)
    # NOTE: the conversion to binary is actually skipped as the resulting
    # integer can just be calculated directly
    def to_int(self):
        value = int(self.type[0]) + 2*int(self.type[1])
        if self.team is None:
            value += 8
        else:
            value += 4*self.team + 16*self.player
        if self.goal is not None:
            return value + 64*self.goal
        return value + 128

    # Decode an integer between 0 and 183 (incl.) to a hit object
    # (see comments on to_int() above)
    def from_int(value):
        if value > 183 or value < 0:
            return None
        if (value >> 7) % 2 == 1:
            goal = None
        else:
            goal = (value >> 6) % 2
        if (value >> 2) % 4 == 2:
            return hit(type=(value % 2 == 1, (value >> 1) % 2 == 1), goal=goal)
        return hit(type=(value % 2 == 1, (value >> 1) % 2 == 1), team=(value >> 2) % 2, player=(value >> 4) % 4, goal=goal)

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

    # Make position relative to a given field center
    def relativize_pos(self, center):
        self.pos = (self.pos[0] - center[0], self.pos[1] - center[1])

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
        self.db = MySQLdb.connect(host="localhost",
                                 user="root",
                                 passwd="password",
                                 db="Tafelvoetbal")

    # Start a game by starting the timer, This method is called automatically
    # in the track.track() function when the ball is detected for the first time
    def start(self):
        self.time = time.time()

    # Stop a game by stopping the timer and removing the final dummy datapoint,
    # This method must be called manually after the game is done
    def stop(self):
        self.duration = time.time() - self.time
        self.datapoints.pop(-1)
        self.relativize()

    # Make all xy coordinates relative to the field center at that time
    def relativize(self):
        prev = 0
        for field_index, field in enumerate(self.fields):
            field.relativize_hull()
            for frame_no, dp in enumerate(self.datapoints[prev:]):
                if dp.field_index != field_index:
                    break
                dp.relativize_pos(field.center)
            prev += frame_no

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

    # Writes game data to SQL database after game is finished
    def write_db(self, zwartachter_ID, zwartvoor_ID, witachter_ID, witvoor_ID):

        cur = self.db.cursor()

        # Convert start time and duration to right format
        db_starttime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.time))
        duration = str(timedelta(seconds=int(self.duration)))

        # Write game data to MySQL database (Table: Games)
        cur.execute("INSERT INTO Games (PlayerID_Black1, PlayerID_Black2, PlayerID_White1, PlayerID_White2,"
                    "StartTime, Duration, ScoreWhite, ScoreBlack) VALUES ({}, {}, {}, {}, '{}', '{}', {}, {})".format(
            zwartachter_ID, zwartvoor_ID, witachter_ID, witvoor_ID, db_starttime, duration, self.score[0], self.score[1]))

        self.db.commit()

        # Fetch current game_ID
        cur.execute("SELECT ID FROM Games ORDER BY ID DESC LIMIT 1")
        game_id = cur.fetchone()[0]

        # Write hulls and datapoints to the database
        prev = 0
        for field_index, field in enumerate(self.fields):
            hull = field.hull_to_string
            cur.execute("INSERT INTO Hulls (Hull) VALUES ('{}')".format(hull))
            hullid = cur.lastrowid

            for frame_no, dp in enumerate(self.datapoints[prev:]):
                if dp.field_index == field_index:
                    x_pos = dp.pos[0]
                    y_pos = dp.pos[1]
                    speed = dp.speed
                    angle = dp.angle
                    accuracy = dp.accuracy

                    prev += 1

                    if dp.hit:
                        print(prev, dp.hit.to_int(), dp.hit.goal)
                        cur.execute(
                            "INSERT INTO Datapoints (FrameNumber, GameID, HullID, XCoord, YCoord, Speed, Angle, Accuracy,"
                            "HitType) VALUES ({}, {}, {}, {}, {}, {}, {}, {}, {})".format(
                                prev, game_id, hullid, x_pos, y_pos, speed, angle, accuracy, dp.hit.to_int()))

                        self.db.commit()

                    else:
                        cur.execute(
                            "INSERT INTO Datapoints (FrameNumber, GameID, HullID, XCoord, YCoord, Speed, Angle, Accuracy)"
                            "VALUES ({}, {}, {}, {}, {}, {}, {}, {})".format(
                                prev, game_id, hullid, x_pos, y_pos, speed, angle, accuracy))

                        self.db.commit()

                else:
                    break