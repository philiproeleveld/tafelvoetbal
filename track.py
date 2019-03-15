import sys
import cv2
import numpy as np
import math



# Constants
source = "game_shorts.mp4"
ballradius = 30
black        = (  0,   0,   0)
blue         = (255,   0,   0)
green        = (  0, 255,   0)
red          = (  0,   0, 255)
cyan         = (255, 255,   0)
pink         = (255,   0, 255)
yellow       = (  0, 255, 255)
white        = (255, 255, 255)
lower_yellow = ( 20, 127, 127)
upper_yellow = ( 35, 255, 255)
lower_blue   = ( 97, 127, 127)
upper_blue   = (107, 255, 255)



# Datapoint definition
class datapoint:
    def __init__(self, pos, speed, angle):
        self.pos = pos
        self.speed = speed
        self.angle = angle



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
        angle = 0
    else:
        angle = math.degrees(math.atan(dy/dx)) + 90
    if dx < 0:
        angle += 180
    return angle

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



# Load video
vid = cv2.VideoCapture(source)
if not vid.isOpened():
    sys.exit(1)



history = [] # List of all datapoints (or None if ball was hidden)
hits = [] # List of all detected hits
last_seen = None # Last location the ball was detected
hidden_timer = 0 # Number of frames in a row where the ball is hidden
score = [0, 0]
framecopy = None # Image of the first frame to display static images



# Loop over all frames
while True:
    ok, frame = vid.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1) # Only use this line when using game_short.mp4

    # Set framecopy to the first frame found
    if framecopy is None:
        framecopy = frame.copy()

    ball_frame = mask_frame(frame, lower_yellow, upper_yellow)

    # Find ball manually based on color
    calc = cv2.resize(ball_frame, (240, 135))
    calc = cv2.split(calc)[2]
    coords = cv2.findNonZero(calc)
    # If enough pixels are found
    if np.size(coords) > 20:
        avg = np.mean(coords, (0, 1)) * 8
        ballcenter = (int(avg[0]), int(avg[1]))
        # If the ball has at all been seen before
        if last_seen:
            # Calculate speed and angle based on last seen and current location
            speed = np.linalg.norm(np.array(last_seen) - np.array(ballcenter)) / (1 + hidden_timer)
            angle = calc_angle(last_seen, ballcenter)
        else:
            speed = 0
            angle = 0
        history.append(datapoint(ballcenter, speed, angle))
        last_seen = ballcenter
        hidden_timer = 0
        cv2.circle(frame, ballcenter, ballradius, blue, 2)
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
        cv2.rectangle(frame, (0, lgoal[1]), (lgoal[0], lgoal[2]), cyan, 2)
        cv2.rectangle(frame, (rgoal[0], rgoal[1]), (1920, rgoal[2]), cyan, 2)

    # Check for goals: if last known location was near a goal
    # and the ball has been gone for 10 frames then a goal was made
    if last_seen and hidden_timer == 10:
        if (last_seen[0] > rgoal[0]) and (last_seen[1] > rgoal[1]) and (last_seen[1] < rgoal[2]):
            score[0] += 1
        if (last_seen[0] < lgoal[0]) and (last_seen[1] > lgoal[1]) and (last_seen[1] < lgoal[2]):
            score[1] += 1

    # Draw recent history
    for i in range(max(1, len(history) - 30), len(history)):
        if history[i]:
            j = i - 1
            while j >= 0 and not history[j]:
                j -= 1
            cv2.line(frame, history[j].pos, history[i].pos, pink, 2)




    # Draw score, speed and angle on frame
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



    # Attempt to determine hits
    if len(history) > 2:
        if history[-2] and history[-1]:
            if abs(history[-2].speed - history[-1].speed) > 30 or (abs(history[-2].angle - history[-1].angle) > 30 and abs((history[-2].angle - history[-1].angle) % 360 - 360) > 30):
                cv2.fillConvexPoly(frame, np.array(((400, 940), (540, 940), (540, 1080), (400, 1080))), red)
                hits.append(history[-2].pos)



    # Fullscreen
    cv2.namedWindow("Tracking", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # Display for at least 1 ms
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()



# Draw "heatmap" on the first frame
heatmap = framecopy.copy()
# Draw entire ball history
first_few = 0
for datapoint in reversed(history):
    if datapoint:
        if first_few < 20:
            color = pink
            alpha = 0.7
        else:
            color = yellow
            alpha = 0.2
        first_few += 1
        temp = heatmap.copy()
        cv2.circle(temp, datapoint.pos, 20, color, -1)
        cv2.addWeighted(temp, alpha, heatmap, 1 - alpha, 0, heatmap)
# Draw all hit locations
first_few = 0
for hit in reversed(hits):
    if first_few < 5:
        color = blue
        alpha = 1
    else:
        color = red
        alpha = 0.3
    first_few += 1
    temp = heatmap.copy()
    cv2.circle(temp, hit, 10, color, -1)
    cv2.addWeighted(temp, alpha, heatmap, 1 - alpha, 0, heatmap)

# Fullscreen
cv2.namedWindow("Heatmap", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Heatmap", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
# Display until Q is pressed
cv2.imshow("Heatmap", heatmap)
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
