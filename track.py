import sys
import cv2
import numpy as np
import math
import cmath
from scipy.spatial import Delaunay
from argparse import ArgumentParser
import game_data

# BGR colors
black        = (  0,   0,   0)
blue         = (255,   0,   0)
green        = (  0, 255,   0)
red          = (  0,   0, 255)
cyan         = (255, 255,   0)
pink         = (255,   0, 255)
yellow       = (  0, 255, 255)
white        = (255, 255, 255)
# HSV colors
lower_yellow = ( 20, 120, 120)
upper_yellow = ( 35, 255, 255)
lower_blue   = (100, 100, 100)
upper_blue   = (110, 255, 255)
lower_green  = ( 70,  70,  50)
upper_green  = (100, 255, 150)

# Field class to remember the current field dimensions, and when to update this
class field_data:
    update_timer = 60
    def __init__(self):
        self.update = 0 # Recalculate field dimensions when this reaches zero
        self.hull = None # Defines the green area of the field
        self.center = None # Coordinates of the center of the field
        self.goals = None # Location of the two goals based on field hull
        self.regions = None # Regions around the player positions (e.g. keeper, midfield)

# Make tracking easier by masking pixels outside the range (lower, upper)
def mask_frame(frame, lower, upper):
    conv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(conv, np.array(lower), np.array(upper))
    conv = cv2.bitwise_and(conv, conv, mask=mask)
    return conv

# Calculate the angle of a trajectory (in degrees)
def calc_angle(point1, point2):
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    if dx == 0:
        if dy < 0:
            return 90
        elif dy > 0:
            return 270
        else:
            return None
    else:
        angle = math.degrees(math.atan(dy/dx))
    if dx < 0:
        angle += 180
    return -angle % 360

# Calculate the most sensical average of a list of angles
# See https://rosettacode.org/wiki/Averages/Mean_angle for an explanation
def mean_angle(angles):
    return math.degrees(cmath.phase(sum(cmath.rect(1, math.radians(angle)) for angle in angles)/len(angles))) % 360



def track(frame, game, field, scaledown, hit_detection=None):
    """
    Functional tracking code: Find the ball in the frame, storing relevant data
    inside the game and field objects.

    arguments:
    frame               frame in which to do the computations
    game                game_data instance containing all data about the ball and game
    field               field_data instance containing the most recent field dimensions
    scaledown           factor to scale down footage by before doing calculations

    optional arguments:
    hit_detection       default: None
        if not none, it will be filled as a tuple of four elements:
        (comparison_speed, speed_thresh, comparison_angle, angle_thresh)
        which can be used to draw the hit detection cone

    returns:
    game                the game_data instance passed to the function filled with new data
    """
    res = (frame.shape[1], frame.shape[0])
    lower_pix_thresh = 100 * (res[0] / (640 * scaledown)) ** 2
    upper_pix_thresh = 400 * (res[0] / (640 * scaledown)) ** 2
    scaled_frame = cv2.resize(frame, (res[0] // scaledown, res[1] // scaledown))

    if field.update == 0:
        # Determine what the field looks like (without sloped/white edges)
        field_frame = mask_frame(scaled_frame, lower_green, upper_green)
        field_frame = cv2.split(field_frame)[2]
        coords = cv2.findNonZero(field_frame)
        if coords is None:
            return
        hull = cv2.convexHull(coords) * scaledown
        field.hull = np.reshape(hull, (np.size(hull) // 2, 2))
        leftmost = field.hull[:, 0].min()
        rightmost = field.hull[:, 0].max()
        fieldwidth = rightmost - leftmost
        topmost = field.hull[:, 1].min()
        botmost = field.hull[:, 1].max()
        fieldheight = botmost - topmost
        # goals is a tuple: (xleft, xright, ytop, ybot) where xleft is the
        # x-coord of the left goal, xright is the x-coord of the right goal,
        # and ytop and ybot determine the vertical range of the two goals
        field.goals = (leftmost, rightmost, topmost + fieldheight // 3, topmost + 2 * fieldheight // 3)
        # Find field center when only looking at the middle part of the frame
        slice_offset = ((leftmost + fieldwidth // 3) // scaledown, field.goals[2] // scaledown)
        sliced_frame = scaled_frame[field.goals[2] // scaledown:field.goals[3] // scaledown, (leftmost + fieldwidth // 3) // scaledown:(leftmost + 2 * fieldwidth // 3) // scaledown, :]
        center_frame = mask_frame(sliced_frame, lower_blue, upper_blue)
        center_frame = cv2.split(center_frame)[2]
        coords = cv2.findNonZero(center_frame)
        if np.size(coords) > lower_pix_thresh:
            avg = np.mean(coords, (0, 1))
            avg += slice_offset
            avg *= scaledown
            field.center = (int(avg[0]), int(avg[1]))
            # Update was successful
            field.update += 1
        else:
            # Use field hull to estimate field center
            field.center = ((leftmost + rightmost) // 2, (topmost + botmost) // 2)
            # An update is required asap
            field.update = 0
        # Field center is used to calculate regions which is a list of the 7
        # x-coords that define vertical lines that divide the 8 player regions
        field.regions = []
        for i in range(4):
            field.regions.append(leftmost + (1 + 2*i) * (field.center[0] - leftmost) // 7)
        for i in range(1, 4):
            field.regions.append(field.center[0] + 2*i * (rightmost - field.center[0]) // 7)
    # Recalculate after some amount of frames specified by the field_data class
    else:
        field.update = (field.update + 1) % field_data.update_timer

    # Find ball based on color
    ball_frame = mask_frame(scaled_frame, lower_yellow, upper_yellow)
    ball_frame = cv2.split(ball_frame)[2]
    coords = cv2.findNonZero(ball_frame)
    # Accuracy of ball detection to be used in determining hit thresholds
    accuracy = min(1, (np.size(coords) - lower_pix_thresh) / (upper_pix_thresh - lower_pix_thresh))
    # If enough pixels are found
    if accuracy > 0:
        avg = np.mean(coords, (0, 1)) * scaledown
        ballcenter = (int(avg[0]), int(avg[1]))
        # If the ball has not been seen before at all
        if not game.last_seen():
            game.start()
            speed = 0
            angle = 0
        else:
            # Predict the location based on game.last_seen()
            dx = game.datapoints[-1].hidden * game.last_seen().speed * math.cos(math.radians(game.last_seen().angle))
            dy = -game.datapoints[-1].hidden * game.last_seen().speed * math.sin(math.radians(game.last_seen().angle))
            prediction = (min(max(0, game.last_seen().pos[0] + dx), res[0]), min(max(0, game.last_seen().pos[1] + dy), res[1]))
            # Note that prediction is equal to game.last_seen().pos if the ball is hidden for 0 frames
            speed = np.linalg.norm(np.array(prediction) - np.array(ballcenter))
            angle = calc_angle(prediction, ballcenter)
            # If the ball hasn't moved at all
            if angle is None:
                # Set angle to last known angle instead
                angle = game.last_seen().angle

            # Determine hits
            # Compare speed only to the first previously known speed
            comparison_speed = game.last_seen().speed
            # Threshold for speed-based hit detection is depends on current speed
            speed_thresh = 100 / (1 +  10 * accuracy ** 2 / 2 ** (comparison_speed / 25))

            # Compare angle to average of all previous angles up to the last hit
            angles = []
            for dp in reversed(game.datapoints[:-1]):
                if dp.angle is not None:
                    angles.append(dp.angle)
                if dp.hit:
                    break
            if angles:
                comparison_angle = mean_angle(angles)
            else:
                comparison_angle = game.last_seen().angle
            # Threshold for angle-based hit detection is determined by speed
            angle_thresh = min(90, 450 / ((comparison_speed * accuracy) ** 0.7 + 3))

            # Save data to draw hit detection
            if hit_detection:
                hit_detection[0] = comparison_speed
                hit_detection[1] = speed_thresh
                hit_detection[2] = comparison_angle
                hit_detection[3] = angle_thresh

            # Detect a hit
            speed_hit = abs(speed - comparison_speed) > speed_thresh
            angle_hit = (comparison_angle - angle) % 360 > angle_thresh and (angle - comparison_angle) % 360 > angle_thresh

            if speed_hit or angle_hit:
                # Outside of green area, assume a side was hit
                if Delaunay(field.hull).find_simplex(game.last_seen().pos) < 0:
                    new_hit = game_data.hit((speed_hit, angle_hit))
                # Inside the field, determine which (row of) player(s) it was
                else:
                    last_xpos = game.last_seen().pos[0]
                    if last_xpos < field.regions[0]:
                        team = game_data.team_black
                        player = game_data.keeper
                    elif last_xpos < field.regions[1]:
                        team = game_data.team_black
                        player = game_data.defense
                    elif last_xpos < field.regions[2]:
                        team = game_data.team_white
                        player = game_data.offense
                    elif last_xpos < field.regions[3]:
                        team = game_data.team_black
                        player = game_data.midfield
                    elif last_xpos < field.regions[4]:
                        team = game_data.team_white
                        player = game_data.midfield
                    elif last_xpos < field.regions[5]:
                        team = game_data.team_black
                        player = game_data.offense
                    elif last_xpos < field.regions[6]:
                        team = game_data.team_white
                        player = game_data.defense
                    else:
                        team = game_data.team_white
                        player = game_data.keeper
                    new_hit = game_data.hit((speed_hit, angle_hit), team=team, player=player)
                game.last_seen().hit = new_hit

        game.datapoints[-1].set_data(ballcenter, speed, angle, accuracy, field.hull, field.center)
        game.add_dp()
    # Couldn't find ball
    else:
        game.datapoints[-1].hidden += 1

    # Check for goals: if last known location was near a goal
    # and the ball has been gone for 10 frames then a goal was made
    if game.datapoints[-1].hidden == 10 and game.last_seen() and game.last_seen().pos[1] > field.goals[2] and game.last_seen().pos[1] < field.goals[3]:
        shot = game.last_seen().pos[0]
        team = None
        player = None
        slowframes = 0
        # Goal at the right side of the field
        if shot > (field.regions[-1] + field.regions[-2]) / 2:
            for i in reversed(range(len(game.datapoints))):
                dp = game.datapoints[i]
                if dp and dp.hit:
                    if dp.speed < 20:
                        slowframes += 1
                        if slowframes == 5:
                            break
                    if dp.pos[0] <= shot:
                        shot = dp.pos[0]
                        team = dp.hit.team
                        player = dp.hit.player
                    else:
                        break
            while True:
                i += 1
                if game.datapoints[i] and game.datapoints[i].hit:
                    break
            game.datapoints[i].hit.goal = game_data.team_white
            if team is None or player == game_data.midfield:
                pass
            elif player == game_data.keeper and team == game_data.team_black:
                game.add_score(1, game_data.team_black)
                game.add_score(-1, game_data.team_white)
            else:
                game.add_score(1, game_data.team_black)
        # Goal at the left side of the field
        elif shot < (field.regions[0] + field.regions[1]) / 2:
            for i in reversed(range(len(game.datapoints))):
                dp = game.datapoints[i]
                if dp and dp.hit:
                    if dp.speed < 20:
                        slowframes += 1
                        if slowframes == 5:
                            break
                    if dp.pos[0] >= shot:
                        shot = dp.pos[0]
                        team = dp.hit.team
                        player = dp.hit.player
                    else:
                        break
            while True:
                i += 1
                if game.datapoints[i] and game.datapoints[i].hit:
                    break
            game.datapoints[i].hit.goal = game_data.team_black
            if team is None or player == game_data.midfield:
                pass
            elif player == game_data.keeper and team == game_data.team_white:
                game.add_score(1, game_data.team_white)
                game.add_score(-1, game_data.team_black)
            else:
                game.add_score(1, game_data.team_white)

    return game



def main(**kwargs):
    """
    Read the video source frame by frame, tracking the ball and displaying
    results as specified by the optional arguments.

    optional arguments:
    arguments       type        default value
    -----------------------------------------
    source          (string)    None
        specifies path to video source, use webcam footage instead if no path

    flip            (boolean)   False
        flip the video source if true

    scaledown       (integer)   1
        specify factor to scale down footage by before doing calculations

    ball            (boolean)   False
        visualize current location of ball if true

    history         (boolean)   False
        visualize recent history if true

    hits            (boolean)   False
        visualize hits if true
    text            (boolean)   False
        show textual data if true
    field           (boolean)   False
        visualize field hull and player regions if true

    goals           (boolean)   False
        visualize goal bounding box if true

    hit_detection   (boolean)   False
        visualize hit detection if true

    heatmap show    (boolean)   False
        show a heatmap after the video is done if true

    wait            (integer)   1
        specify how long to wait on each frame (in ms)
    """
    # Unpack kwargs
    source = kwargs['source'] if 'source' in kwargs and kwargs['source'] else None
    flip = kwargs['flip'] if 'flip' in kwargs else False
    scaledown = kwargs['scaledown'] if 'scaledown' in kwargs else 1
    draw_ball = kwargs['ball'] if 'ball' in kwargs else False
    draw_history = kwargs['history'] if 'history' in kwargs else False
    draw_hits = kwargs['hits'] if 'hits' in kwargs else False
    draw_text = kwargs['text'] if 'text' in kwargs else False
    draw_field = kwargs['field'] if 'field' in kwargs else False
    draw_goals = kwargs['goals'] if 'goals' in kwargs else False
    draw_hit_detection = kwargs['hit_detection'] if 'hit_detection' in kwargs else False
    draw_heatmap = kwargs['heatmap'] if 'heatmap' in kwargs else False
    wait = kwargs['wait'] if 'wait' in kwargs else 1

    # Load video
    if source:
        vid = cv2.VideoCapture(source)
    else:
        vid = cv2.VideoCapture(0)
    if not vid.isOpened():
        sys.exit(1)

    # Read first frame to determine resolution
    ok, frame = vid.read()
    if not ok:
        sys.exit(1)

    # Constants based on resolution
    res = (frame.shape[1], frame.shape[0])
    scale = res[0] / 640
    draw_thickness = max(1, int(scale) - 1)
    ballradius = int(10 * scale)

    # Variables
    game = game_data.game_data()
    field = field_data()
    thumbnail = None # Image of the first frame to display static images

    # Loop over all frames
    while True:
        ok, frame = vid.read()
        if not ok:
            break

        if flip:
            frame = cv2.flip(frame, 1)

        # Declare some variables to draw hit detection later
        if draw_hit_detection:
            hit_detection = [0, 0, 0, 0]
        else:
            hit_detection = None

        # Set thumbnail to the first frame found
        if draw_heatmap and thumbnail is None:
            thumbnail = frame.copy()

        track(frame, game, field, scaledown, hit_detection=hit_detection)

        ###################################################################
        # End of functional code, the following is only for visualization #
        ###################################################################

        # Draw ball tracking
        if game.last_seen():
            if draw_ball:
                cv2.circle(frame, game.last_seen().pos, ballradius, blue, draw_thickness)
            if draw_hit_detection:
                # Draw area between the thresholds on frame
                pos = game.last_seen().pos
                outerpts = cv2.ellipse2Poly(pos, (int(max(0, hit_detection[0] - hit_detection[1])), int(max(0, hit_detection[0] - hit_detection[1]))), -int(hit_detection[2]), -int(hit_detection[3]), int(hit_detection[3]), 2)
                innerpts = cv2.ellipse2Poly(pos, (int(hit_detection[0] + hit_detection[1]), int(hit_detection[0] + hit_detection[1])), -int(hit_detection[2]), -int(hit_detection[3]), int(hit_detection[3]), 2)
                pts = np.vstack((outerpts, innerpts[::-1]))
                cv2.fillPoly(frame, [pts], green)

        # Draw recent history and/or hits
        if draw_history or draw_hits:
            for i in range(max(1, len(game.datapoints[:-1]) - 100), len(game.datapoints[:-1])):
                # Also draw hits where applicable
                if draw_hits and game.datapoints[i].hit:
                    temp = frame.copy()
                    if game.datapoints[i].hit.team == game_data.team_black:
                        color = black
                    elif game.datapoints[i].hit.team == game_data.team_white:
                        color = white
                    else:
                        color = red
                    cv2.circle(temp, game.datapoints[i].pos, ballradius // 2, color, -1)
                    alpha = 0.5
                    cv2.addWeighted(temp, alpha, frame, 1 - alpha, 0, frame)
                if draw_history:
                    if game.datapoints[i-1]:
                        cv2.line(frame, game.datapoints[i-1].pos, game.datapoints[i].pos, pink, draw_thickness)

        # Draw field hull and player regions
        if draw_field:
            cv2.polylines(frame, [field.hull], True, red, draw_thickness, cv2.LINE_8)
            for region in field.regions:
                cv2.line(frame, (region, 0), (region, res[1]), red, draw_thickness)

        # Draw goal rectangles
        if draw_goals:
            cv2.rectangle(frame, (0, field.goals[2]), ((field.regions[0] + field.regions[1]) // 2, field.goals[3]), cyan, draw_thickness)
            cv2.rectangle(frame, ((field.regions[-1] + field.regions[-2]) // 2, field.goals[2]), (res[0], field.goals[3]), cyan, draw_thickness)

        # Draw score, speed and angle on frame
        if draw_text:
            # White backdrop to make text easier to read
            cv2.fillConvexPoly(frame, np.array(((0, int(res[1] - 80 * scale)), (int(200 * scale), int(res[1] - 80 * scale)), (int(200 * scale), res[1]), (0, res[1]))), white)
            # Score
            cv2.putText(frame, "Score: " + str(game.score[0]) + " - " + str(game.score[1]), (int(3 * scale), int(res[1] - 60 * scale)), cv2.FONT_HERSHEY_SIMPLEX, scale / 2, black, draw_thickness)
            # Speed and angle
            if game.last_seen():
                cv2.putText(frame, "Speed: " + str(int(game.last_seen().speed)) + " pix/frame", (int(3 * scale), int(res[1] - 34 * scale)), cv2.FONT_HERSHEY_SIMPLEX, scale / 2, black, draw_thickness)
                cv2.putText(frame, "Angle: " + str(int(game.last_seen().angle)) + " deg", (int(3 * scale), int(res[1] - 8 * scale)), cv2.FONT_HERSHEY_SIMPLEX, scale / 2, black, draw_thickness)

        # Fullscreen
        cv2.namedWindow("Tracking", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # Display
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(wait) & 0xFF == ord('q'):
            if wait > 0:
                break

    game.stop()
    cv2.destroyAllWindows()

    # Draw "heatmap" on the first frame
    if draw_heatmap:
        heatmap = thumbnail.copy()
        # Draw entire ball history
        first_few = 0
        for dp in reversed(game.datapoints[:-1]):
            if first_few < 20:
                color = pink
                alpha = 0.7
            else:
                color = yellow
                alpha = 0.2
            first_few += 1
            temp = heatmap.copy()
            cv2.circle(temp, dp.pos, 20, color, -1)
            cv2.addWeighted(temp, alpha, heatmap, 1 - alpha, 0, heatmap)
            # Also draw hits
            if draw_hits and dp.hit:
                temp = heatmap.copy()
                if dp.hit.team == game_data.team_black:
                    color = black
                elif dp.hit.team == game_data.team_white:
                    color = white
                else:
                    color = red
                cv2.circle(temp, dp.pos, ballradius // 2, color, -1)
                alpha = 0.5
                cv2.addWeighted(temp, alpha, heatmap, 1 - alpha, 0, heatmap)

        # Fullscreen
        cv2.namedWindow("Heatmap", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Heatmap", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # Display until Q is pressed
        cv2.imshow("Heatmap", heatmap)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()



if __name__ == '__main__':
    # Configure ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-s", "--source", metavar="SRC", help="specify path to video source")
    parser.add_argument("--flip", action="store_true", help="flip the video source")
    parser.add_argument("--scaledown", type=int, default=1, metavar="N", help="specify how much to scale down footage before doing calculations")
    parser.add_argument("--ball", action="store_true", help="turn on ball tracking visualization")
    parser.add_argument("--history", action="store_true", help="turn on recent history visualization")
    parser.add_argument("--hits", action="store_true", help="turn on hit visualization")
    parser.add_argument("--text", action="store_true", help="turn on textual data")
    parser.add_argument("--field", action="store_true", help="turn on field bounding box visualization")
    parser.add_argument("--goals", action="store_true", help="turn on goal bounding box visualization")
    parser.add_argument("--hit-detection", action="store_true", help="turn on hit detection visualization")
    parser.add_argument("--heatmap", action="store_true", help="show a heatmap after the video is done")
    parser.add_argument("--wait", type=int, default=1, metavar="N", help="specify how long to wait on each frame")
    args = parser.parse_args()

    main(**vars(args))
