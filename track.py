import sys
import cv2
import numpy as np
import math
import cmath
from scipy.spatial import Delaunay
from argparse import ArgumentParser



# Constants
field_update_timer = 60
color_detect_min = 150
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
lower_yellow = ( 20, 120, 120)
upper_yellow = ( 35, 255, 255)
lower_blue   = (100, 100, 100)
upper_blue   = (110, 255, 255)
lower_green  = ( 70,  70,  50)
upper_green  = (100, 255, 150)



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



# Configure ArgumentParser
parser = ArgumentParser()
parser.add_argument("-s", "--source", metavar="SRC", help="specify video source location")
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
parser.add_argument("--wait-on-hits", type=int, default=-1, metavar="N", help="specify how long to wait on frames where a hit occurs")
parser.add_argument("--scaledown", type=int, default=1, metavar="N", help="specify how much to scale down footage before doing calculations")
args = parser.parse_args()
if args.wait_on_hits < 0:
    args.wait_on_hits = args.wait



# Load video
if args.source:
    vid = cv2.VideoCapture(args.source)
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
score = [0, 0]
history = [] # List of all datapoints (None value if ball was hidden)
hits = [] # List of all detected hits
last_seen = None # Last location the ball was detected
hidden_timer = 0 # Number of frames in a row where the ball is hidden
hit_timer = 0 # Number of frames in a row since the last hit
field_update = 0 # Recalculate field dimensions when this reaches zero
hull = None # Defines the green area of the field
lgoal = None # Location of the left goal based on field hull
rgoal = None # Location of the right goal based on field hull
regions = None # Location of center field
fieldwidth = None # Width of the field based on the two goals
framecopy = None # Image of the first frame to display static images



# Loop over all frames
while True:
    # Wait will be set to args.wait_on_hits if a hit occurs
    wait_time = args.wait

    ok, frame = vid.read()
    if not ok:
        break
    scaled_frame = cv2.resize(frame, (res[0] // args.scaledown, res[1] // args.scaledown))

    if args.flip:
        frame = cv2.flip(frame, 1)

    # Set framecopy to the first frame found
    if args.heatmap:
        if framecopy is None:
            framecopy = frame.copy()

    field_update += 1
    if field_update == 1:
        # Determine what the field looks like (without sloped/white edges)
        field_frame = mask_frame(scaled_frame, lower_green, upper_green)
        field_frame = cv2.split(field_frame)[2]
        coords = cv2.findNonZero(field_frame)
        if coords is None:
            break
        hull = cv2.convexHull(coords) * args.scaledown
        hull = np.reshape(hull, (np.size(hull) // 2, 2))
        leftmost = hull[:, 0].min()
        rightmost = hull[:, 0].max()
        fieldwidth = rightmost - leftmost
        # Find goals based on field hull
        lgoal = hull[hull[:, 0] < leftmost + fieldwidth // 30]
        rgoal = hull[hull[:, 0] > rightmost - fieldwidth // 30]
        # each goal is a tuple: (x, ytop, ybot)  where ytop and ybot are the
        # vertical top and bottom of the goal and x is the horizontal location
        lgoal = (leftmost, lgoal[:, 1].min(), lgoal[:, 1].max())
        rgoal = (rightmost, rgoal[:, 1].min(), rgoal[:, 1].max())
        # Find center field
        slice_offset = ((leftmost + fieldwidth // 3) // args.scaledown, (lgoal[1] + rgoal[1]) // (2 * args.scaledown))
        sliced_frame = scaled_frame[(lgoal[1] + rgoal[1]) // (2 * args.scaledown):(lgoal[2] + rgoal[2]) // (2 * args.scaledown), (leftmost + fieldwidth // 3) // args.scaledown:(leftmost + 2 * fieldwidth // 3) // args.scaledown, :]
        center_frame = mask_frame(sliced_frame, lower_blue, upper_blue)
        center_frame = cv2.split(center_frame)[2]
        coords = cv2.findNonZero(center_frame)
        if np.size(coords) > color_detect_min * (scale / args.scaledown) ** 2:
            avg = np.mean(coords, (0, 1))
            avg += slice_offset
            avg *= args.scaledown
            fieldcenter = (int(avg[0]), int(avg[1]))
            lwidth = fieldcenter[0] - lgoal[0]
            rwidth = rgoal[0] - fieldcenter[0]
            regions = []
            for i in range(4):
                regions.append(lgoal[0] + (1 + 2*i) * lwidth // 7)
            for i in range(1, 4):
                regions.append(fieldcenter[0] + 2*i * rwidth // 7)
        else:
            lwidth = (rgoal[0] - lgoal[0]) // 2
            rwidth = (rgoal[0] - lgoal[0]) // 2
            regions = [lgoal[0] + (1 + 2*i) * lwidth // 7 for i in range(7)]
            field_update = 0
    # Recalculate after some amount of frames
    elif field_update == field_update_timer:
        field_update = 0

    # Find ball based on color
    ball_frame = mask_frame(scaled_frame, lower_yellow, upper_yellow)
    ball_frame = cv2.split(ball_frame)[2]
    coords = cv2.findNonZero(ball_frame)
    # If enough pixels are found
    if np.size(coords) > color_detect_min * (scale / args.scaledown) ** 2:
        avg = np.mean(coords, (0, 1)) * args.scaledown
        ballcenter = (int(avg[0]), int(avg[1]))
        # If the ball has not been seen before at all
        if not last_seen:
            speed = 0
            angle = 0
            lower_speed_thresh = 0
            upper_speed_thresh = 10
            comparison_angle = 0
            angle_thresh = 90
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
            lower_speed_thresh = max(0, 0.5 * comparison_speed - 10)
            upper_speed_thresh = 1.5 * comparison_speed + 10

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

            # Remember what kind of hit was detected
            speed_hit = speed < lower_speed_thresh or speed > upper_speed_thresh
            angle_hit = (comparison_angle - angle) % 360 > angle_thresh and (angle - comparison_angle) % 360 > angle_thresh

            if speed_hit or angle_hit:
                # Outside of green area, assume a side was hit
                if Delaunay(hull).find_simplex(last_seen.pos) < 0:
                    hits.append(hit(last_seen))
                # Inside the field, determine which player it was
                else:
                    print(last_seen.pos[0], regions)
                    for i in range(len(regions)):
                        if last_seen.pos[0] < regions[i]:
                            if i == 0:
                                team = team_black
                                player = keeper
                            elif i == 1:
                                team = team_black
                                player = defense
                            elif i == 2:
                                team = team_white
                                player = offense
                            elif i == 3:
                                team = team_black
                                player = midfield
                            elif i == 4:
                                team = team_white
                                player = midfield
                            elif i == 5:
                                team = team_black
                                player = offense
                            elif i == 6:
                                team = team_white
                                player = defense
                            else:
                                team = team_white
                                player = keeper
                            hits.append(hit(last_seen, team=team, player=player))
                            break
                    # TODO - determine which team hit the ball.
                hit_timer = 0
                wait_time = args.wait_on_hits
            else:
                hit_timer += 1

        last_seen = datapoint(ballcenter, speed, angle)
        history.append(last_seen)
        hidden_timer = 0
    else:
        history.append(None)
        hidden_timer += 1

    # Check for goals: if last known location was near a goal
    # and the ball has been gone for 10 frames then a goal was made
    if last_seen and hidden_timer == 10:
        if (last_seen.pos[0] > rgoal[0] - fieldwidth // 14) and (last_seen.pos[1] > rgoal[1]) and (last_seen.pos[1] < rgoal[2]):
            score[0] += 1
        if (last_seen.pos[0] < lgoal[0] + fieldwidth // 14) and (last_seen.pos[1] > lgoal[1]) and (last_seen.pos[1] < lgoal[2]):
            score[1] += 1

    # Draw ball tracking
    if last_seen:
        if args.ball:
            cv2.circle(frame, last_seen.pos, ballradius, blue, draw_thickness)
        if args.hit_detection:
            # Draw area between the thresholds on frame
            outerpts = cv2.ellipse2Poly(last_seen.pos, (int(speedvect_scalar * lower_speed_thresh), int(speedvect_scalar * lower_speed_thresh)), -int(comparison_angle), -int(angle_thresh), int(angle_thresh), 2)
            innerpts = cv2.ellipse2Poly(last_seen.pos, (int(speedvect_scalar * upper_speed_thresh), int(speedvect_scalar * upper_speed_thresh)), -int(comparison_angle), -int(angle_thresh), int(angle_thresh), 2)
            pts = np.vstack((outerpts, innerpts[::-1]))
            cv2.fillPoly(frame, [pts], green)
            dx = int(speedvect_scalar * last_seen.speed * math.cos(math.radians(angle)))
            dy = -int(speedvect_scalar * last_seen.speed * math.sin(math.radians(angle)))
            cv2.arrowedLine(frame, last_seen.pos, (last_seen.pos[0] + dx, last_seen.pos[1] + dy), blue, draw_thickness)


    # Draw recent history
    if args.history:
        for i in range(max(1, len(history) - 30), len(history)):
            if history[i]:
                j = i - 1
                while j > 0 and not history[j]:
                    j -= 1
                if history[j]:
                    cv2.line(frame, history[j].pos, history[i].pos, pink, draw_thickness)

    # Draw recent hits
    if args.hits:
        for i in range(max(1, len(hits) - 5), len(hits)):
            temp = frame.copy()
            cv2.circle(temp, hits[i].dp.pos, ballradius // 2, hits[i].get_color(), -1)
            alpha = 0.5
            cv2.addWeighted(temp, alpha, frame, 1 - alpha, 0, frame)

    # Draw field hull and player regions
    if args.field:
        cv2.polylines(frame, [hull], True, red, draw_thickness, cv2.LINE_8)
        for region in regions:
            cv2.line(frame, (region, 0), (region, res[1]), red, draw_thickness)

    # Draw goal rectangles
    if args.goals:
        cv2.rectangle(frame, (0, lgoal[1]), (lgoal[0] + lwidth // 7, lgoal[2]), cyan, draw_thickness)
        cv2.rectangle(frame, (rgoal[0] - rwidth // 7, rgoal[1]), (res[0], rgoal[2]), cyan, draw_thickness)

    # Draw score, speed and angle on frame
    if args.text:
        # White backdrop to make text easier to read
        cv2.fillConvexPoly(frame, np.array(((0, int(res[1] - 80 * scale)), (int(200 * scale), int(res[1] - 80 * scale)), (int(200 * scale), res[1]), (0, res[1]))), white)
        # Score
        cv2.putText(frame, "Score: " + str(score[0]) + " - " + str(score[1]), (int(3 * scale), int(res[1] - 60 * scale)), cv2.FONT_HERSHEY_SIMPLEX, scale / 2, black, draw_thickness)
        # Speed and angle
        if history[-1]:
            cv2.putText(frame, "Speed: " + str(int(history[-1].speed)) + " pix/frame", (int(3 * scale), int(res[1] - 34 * scale)), cv2.FONT_HERSHEY_SIMPLEX, scale / 2, black, draw_thickness)
            cv2.putText(frame, "Angle: " + str(int(history[-1].angle)) + " deg", (int(3 * scale), int(res[1] - 8 * scale)), cv2.FONT_HERSHEY_SIMPLEX, scale / 2, black, draw_thickness)
        else:
            cv2.putText(frame, "Speed: ??? pix/frame", (int(3 * scale), int(res[1] - 34 * scale)), cv2.FONT_HERSHEY_SIMPLEX, scale / 2, black, draw_thickness)
            cv2.putText(frame, "Angle: ??? deg", (int(3 * scale), int(res[1] - 8 * scale)), cv2.FONT_HERSHEY_SIMPLEX, scale / 2, black, draw_thickness)

    # Fullscreen
    cv2.namedWindow("Tracking", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # Display
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        if wait_time > 0:
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
            cv2.circle(temp, hit.dp.pos, ballradius // 2, hit.get_color(), -1)
            cv2.addWeighted(temp, 0.5, heatmap, 0.5, 0, heatmap)

    # Fullscreen
    cv2.namedWindow("Heatmap", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Heatmap", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # Display until Q is pressed
    cv2.imshow("Heatmap", heatmap)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
