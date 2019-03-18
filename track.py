import sys
import cv2
import numpy as np
import math
import cmath
from scipy.spatial import Delaunay
from argparse import ArgumentParser



# Constants
ballradius = 30
speedvect_scalar = 2
team_black = 0
team_white = 1
keeper   = 0
defense  = 1
midfield = 2
offense  = 3
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
lower_yellow = ( 20, 127, 127)
upper_yellow = ( 35, 255, 255)
lower_blue   = ( 97, 127, 127)
upper_blue   = (107, 255, 255)
lower_green1 = ( 75, 150,  50)
lower_green2 = ( 75,  10, 150)
upper_green  = ( 85, 255, 255)



# Datapoint definition
class datapoint:
    def __init__(self, pos, speed, angle):
        self.pos = pos
        self.speed = speed
        self.angle = angle

# Hit definition
class hit:
    def __init__(self, dp, team=None, player=None):
        self.dp = dp
        self.team = team
        self.player = player

    def get_color(self):
        if self.team == team_black:
            return black
        elif self.team == team_white:
            return white
        else:
            return red



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

# Initialize a tracker
def init_tracker(frame, coords):
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, (
            coords[0] - ballradius,
            coords[1] - ballradius,
            ballradius * 2,
            ballradius * 2
    ))
    return tracker



# Configure ArgumentParser
parser = ArgumentParser()
parser.add_argument("-s", "--source", default="game.mp4", metavar="SRC", help="specify video source location")
parser.add_argument("--flip", action="store_true", help="flip the video source")
parser.add_argument("--ball", action="store_false", help="turn OFF ball tracking visualization")
parser.add_argument("--history", action="store_false", help="turn OFF recent history visualization")
parser.add_argument("--hits", action="store_false", help="turn OFF hit visualization")
parser.add_argument("--text", action="store_false", help="turn OFF textual data")
parser.add_argument("--field", action="store_true", help="turn ON field bounding box visualization")
parser.add_argument("--goals", action="store_true", help="turn ON goal bounding box visualization")
parser.add_argument("--hit-detection", action="store_true", help="turn ON hit detection visualization")
parser.add_argument("--heatmap", action="store_true", help="show a heatmap after the video is done")
parser.add_argument("--wait", type=int, default=1, metavar="N", help="specify how long to wait on each frame")
parser.add_argument("--wait-on-hits", type=int, default=0, metavar="N", help="specify how long to wait on frames where a hit occurs")
args = parser.parse_args()
if args.wait_on_hits == 0:
    args.wait_on_hits = args.wait


# Load video
vid = cv2.VideoCapture(args.source)
if not vid.isOpened():
    sys.exit(1)



score = [0, 0]
history = [] # List of all datapoints (None value if ball was hidden)
hits = [] # List of all detected hits
last_seen = None # Last location the ball was detected
hidden_timer = 0 # Number of frames in a row where the ball is hidden
hit_timer = 0 # Number of frames in a row since the last hit
framecopy = None # Image of the first frame to display static images



# Loop over all frames
while True:
    wait_time = args.wait

    ok, frame = vid.read()
    if not ok:
        break

    if args.flip:
        frame = cv2.flip(frame, 1) # Only use this line when using game_short.mp4

    # Set framecopy to the first frame found
    if args.heatmap:
        if framecopy is None:
            framecopy = frame.copy()

    # Determine where the field is (without sloped/white edges)
    ball_frame = mask_frame(frame, lower_yellow, upper_yellow)
    field1 = mask_frame(frame, lower_green1, upper_green)
    field2 = mask_frame(frame, lower_green2, upper_green)
    field = cv2.bitwise_or(field1, field2)
    calc = cv2.resize(field, (240, 135))
    calc = cv2.split(calc)[2]
    coords = cv2.findNonZero(calc)
    hull = cv2.convexHull(coords) * 8

    # Find ball manually based on color
    calc = cv2.resize(ball_frame, (240, 135))
    calc = cv2.split(calc)[2]
    coords = cv2.findNonZero(calc)

    # If enough pixels are found
    if np.size(coords) > 20:
        avg = np.mean(coords, (0, 1)) * 8
        ballcenter = (int(avg[0]), int(avg[1]))
        # If the ball has not been seen before at all
        if not last_seen:
            speed = 0
            angle = 0
        else:
            # Calculate speed and angle based on last seen and current location
            speed = np.linalg.norm(np.array(last_seen.pos) - np.array(ballcenter)) / (1 + hidden_timer)
            angle = calc_angle(last_seen.pos, ballcenter)
            # If the ball hasn't moved at all
            if angle is None:
                # Set angle to last known angle instead
                angle = last_seen.angle

            # Determine hits
            # Compare speed only to the first previously known speed
            comparison_speed = last_seen.speed
            # Threshold for speed-based hit detection is depends on current speed
            lower_speed_thresh = max(0, comparison_speed - (10 + 0.3 * speed))
            upper_speed_thresh = comparison_speed + 10 + 0.3 * speed

            # Compare angle to average of all previous angles up to the last hit
            angles = []
            for dp in history[-(1 + hit_timer):]:
                if dp:
                    if dp.angle is not None:
                        angles.append(dp.angle)
            if angles:
                comparison_angle = mean_angle(angles)
            else:
                comparison_angle = last_seen.angle
            # Threshold for angle-based hit detection is determined by speed
            angle_thresh = min(90, 450 / (speed ** 0.7 + 3))

            # Draw area between the thresholds on frame
            if args.hit_detection:
                outerpts = cv2.ellipse2Poly(ballcenter, (int(speedvect_scalar * lower_speed_thresh), int(speedvect_scalar * lower_speed_thresh)), -int(comparison_angle), -int(angle_thresh), int(angle_thresh), 2)
                innerpts = cv2.ellipse2Poly(ballcenter, (int(speedvect_scalar * upper_speed_thresh), int(speedvect_scalar * upper_speed_thresh)), -int(comparison_angle), -int(angle_thresh), int(angle_thresh), 2)
                pts = np.vstack((outerpts, innerpts[::-1]))
                cv2.fillPoly(frame, [pts], green)

            # Remember what kind of hit was detected
            speed_hit = speed < lower_speed_thresh or speed > upper_speed_thresh
            angle_hit = (comparison_angle - angle) % 360 > angle_thresh and (angle - comparison_angle) % 360 > angle_thresh

            if speed_hit or angle_hit:
                if args.hits:
                    # The ball is outside of the green area
                    if Delaunay(np.reshape(hull, (np.size(hull) // 2, 2))).find_simplex(last_seen.pos) < 0:
                        color = pink
                    elif speed_hit and angle_hit:
                        color = green
                    elif speed_hit:
                        color = red
                    else:
                        color = blue
                    if args.hit_detection:
                        if args.text:
                            cv2.fillConvexPoly(frame, np.array(((400, 940), (540, 940), (540, 1080), (400, 1080))), color)
                        else:
                            cv2.fillConvexPoly(frame, np.array(((0, 940), (140, 940), (140, 1080), (0, 1080))), color)
                else:
                    color = None
                hits.append(hit(last_seen))
                # TODO - determine which team hit the ball.
                hit_timer = 0
                wait_time = args.wait_on_hits
            else:
                hit_timer += 1

        last_seen = datapoint(ballcenter, speed, angle)
        history.append(last_seen)
        hidden_timer = 0
        if args.ball:
            cv2.circle(frame, ballcenter, ballradius, blue, 2)
        if args.hit_detection:
            dx = int(speedvect_scalar * speed * math.cos(math.radians(angle)))
            dy = -int(speedvect_scalar * speed * math.sin(math.radians(angle)))
            cv2.arrowedLine(frame, ballcenter, (ballcenter[0] + dx, ballcenter[1] + dy), blue, 2)
    else:
        history.append(None)
        hidden_timer += 1

    # Find the goals based on distance to the blue center of the field
    goal_frame = mask_frame(frame, lower_blue, upper_blue)
    calc = cv2.resize(goal_frame, (240, 135))
    calc = cv2.split(calc)[2]
    coords = cv2.findNonZero(calc)
    # If enough pixels are found
    if np.size(coords) > 20:
        avg = np.mean(coords, (0, 1)) * 8
        fieldcenter = (int(avg[0]), int(avg[1]))
        # Relative locations are hard-coded for now
        lgoal = (
                max(140, fieldcenter[0] - 680),
                fieldcenter[1] - 175,
                fieldcenter[1] + 189
        )
        rgoal = (
                min(1780, fieldcenter[0] + 820),
                fieldcenter[1] - 180,
                fieldcenter[1] + 246
        )
        if args.goals:
            cv2.rectangle(frame, (0, lgoal[1]), (lgoal[0], lgoal[2]), cyan, 2)
            cv2.rectangle(frame, (rgoal[0], rgoal[1]), (1920, rgoal[2]), cyan, 2)

    # Check for goals: if last known location was near a goal
    # and the ball has been gone for 10 frames then a goal was made
    if last_seen and hidden_timer == 10:
        if (last_seen.pos[0] > rgoal[0]) and (last_seen.pos[1] > rgoal[1]) and (last_seen.pos[1] < rgoal[2]):
            score[0] += 1
        if (last_seen.pos[0] < lgoal[0]) and (last_seen.pos[1] > lgoal[1]) and (last_seen.pos[1] < lgoal[2]):
            score[1] += 1



    # Draw recent history
    if args.history:
        for i in range(max(1, len(history) - 30), len(history)):
            if history[i]:
                j = i - 1
                while j >= 0 and not history[j]:
                    j -= 1
                cv2.line(frame, history[j].pos, history[i].pos, pink, 2)

    # Draw recent hits
    if args.hits:
        for i in range(max(1, len(hits) - 5), len(hits)):
            temp = frame.copy()
            cv2.circle(temp, hits[i].dp.pos, 10, hits[i].get_color(), -1)
            alpha = 0.5
            cv2.addWeighted(temp, alpha, frame, 1 - alpha, 0, frame)

    # Draw score, speed and angle on frame
    if args.text:
        # White backdrop to make text easier to read
        cv2.fillConvexPoly(frame, np.array(((0, 940), (400, 940), (400, 1080), (0, 1080))), white)
        # Score
        cv2.putText(frame, "Score: " + str(score[0]) + " - " + str(score[1]), (10, 980), cv2.FONT_HERSHEY_SIMPLEX, 1, black, 2)
        # Speed and angle
        if history[-1]:
            cv2.putText(frame, "Speed: " + str(int(history[-1].speed)) + " pix/frame", (10, 1020), cv2.FONT_HERSHEY_SIMPLEX, 1, black, 2)
            cv2.putText(frame, "Angle: " + str(int(history[-1].angle)) + " deg", (10, 1060), cv2.FONT_HERSHEY_SIMPLEX, 1, black, 2)
        else:
            cv2.putText(frame, "Speed: ??? pix/frame", (10, 1020), cv2.FONT_HERSHEY_SIMPLEX, 1, black, 2)
            cv2.putText(frame, "Angle: ??? deg", (10, 1060), cv2.FONT_HERSHEY_SIMPLEX, 1, black, 2)

    # Draw field hull
    if args.field:
        cv2.polylines(frame, [hull], True, red, 2, cv2.LINE_8)

    # Fullscreen
    cv2.namedWindow("Tracking", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # Display
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()



# Draw "heatmap" on the first frame
if args.heatmap:
    heatmap = framecopy.copy()
    # Draw entire ball history
    first_few = 0
    for dp in reversed(history):
        if dp:
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

    # Draw all hit locations
    if args.hits:
        for hit in hits:
            temp = heatmap.copy()
            cv2.circle(temp, hit.dp.pos, 10, hit.get_color(), -1)
            cv2.addWeighted(temp, 0.5, heatmap, 0.5, 0, heatmap)

    # Fullscreen
    cv2.namedWindow("Heatmap", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Heatmap", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # Display until Q is pressed
    cv2.imshow("Heatmap", heatmap)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
