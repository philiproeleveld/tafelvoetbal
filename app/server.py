from flask import Flask, render_template, redirect, request, flash, url_for
import cv2
import numpy as np
import MySQLdb
import time
import json
from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))) + "\\track")
from track import track
from game_data import game_data

app = Flask(__name__)
app.secret_key = 'cookie'

# Setup database connection
db = MySQLdb.connect(host="localhost",
                     user="root",
                     passwd="DigitalPower01",
                     db="Tafelvoetbal")

cur = db.cursor()

# GAME GLOBALS (set to None or overwritten for each new game)
wit_voor = None
wit_achter = None
zwart_voor = None
zwart_achter = None
witvoor_ID = None
witachter_ID = None
zwartvoor_ID = None
zwartachter_ID = None
start_time = None
duration = None
score = [0, 0]
video_camera = None
game = None # Contains the game data
game_running = False # True if there is currently a game being played

# Render the three web templates
@app.route('/')
def index():
    # Login page includes all current usernames in database
    cur.execute("SELECT Username FROM Players")
    all_usernames = [row[0] for row in cur.fetchall()]

    return render_template('index.html', db_names=all_usernames)

@app.route('/game')
def login():
    return render_template('game.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/index_redirect', methods=['POST'])
def index_redirect():
    # Login page includes all current usernames in database
    cur.execute("SELECT Username FROM Players")
    all_usernames = [row[0] for row in cur.fetchall()]

    return render_template('index.html', db_names=all_usernames)

@app.route('/handle_register', methods=['POST'])
def handle_register():
    """
    Handles registration of users when the register button is clicked in register.html.
    It flashes an error on the registration webpage when the requirements are not met, and
    writes the registration to MySQL database (table: Players) when the requirements for registration
    are met, after which the login-page template is rendered.

    returns:
    redirect(url_for('register'))                           if requirements for registration are not met
    render_template('index.html', db_names=all_usernames)   if requirements are met
    """

    # Acquire form variables from filled in register.html
    username = request.form['username']
    name = request.form['name']
    surname = request.form['surname']
    age = request.form['age']
    gender = request.form['gender']
    occupation = request.form['function']

    # Acquire usernames in players table for username check
    cur.execute("SELECT username FROM Players WHERE Username = %s", (username,))
    db_usernames = cur.fetchall()

    # Check if conditions for filling in register form are met
    if len(username) == 0 or len(age) == 0:
        error = 'Vul de verplichte velden in'
    elif db_usernames:
        error = 'Username al in gebruik'
    elif age.isdigit() == False:
        error = "Leeftijd dient een getal te zijn"
    elif int(age) > 70:
        error = 'Leeftijd mag niet hoger zijn dan 70'
    elif int(age) < 18:
        error = 'Leeftijd mag niet lager zijn dan 18'

    # Flash message on register page to user when there is an error
    if 'error' in locals():
        flash(error)
        return redirect(url_for('register'))

    # Else write register data to SQL database and return to login page
    else:
        if len(name) == 0 and len(surname) > 0:
            cur.execute("INSERT INTO Players (Username, LastName, Age, Gender, Occupation) VALUES ('{}', '{}', {}, '{}', '{}')".format(
                        username, surname, int(age), gender, occupation))
        elif len(surname) == 0 and len(name) > 0:
            cur.execute("INSERT INTO Players (Username, FirstName, Age, Gender, Occupation) VALUES ('{}', '{}', {}, '{}', '{}')".format(
                        username, name, int(age), gender, occupation))
        elif len(name) == 0 and len(surname) == 0:
            cur.execute("INSERT INTO Players (Username, Age, Gender, Occupation) VALUES ('{}', {}, '{}', '{}')".format(
                        username, int(age), gender, occupation))
        else:
            cur.execute("INSERT INTO Players (Username, FirstName, LastName, Age, Gender, Occupation) VALUES ('{}', '{}', '{}', {}, '{}', '{}')".format(
                        username, name, surname, int(age), gender, occupation))

        db.commit()

        # Create username list to pass login page for selector buttons
        query = cur.execute("SELECT Username FROM Players")
        all_usernames = [row[0] for row in cur.fetchall()]

        return render_template('index.html', db_names=all_usernames)

@app.route('/handle_login', methods=['POST'])
def handle_login():
   """
   Handles the start of a game when login button is clicked in index.html. Upon start of the game
   the user-ID's that match to the usernames are retrieved from the MySQL database (table: Players) and
   are saved in the global environment. After this, game.html is rendered. When this template is rendered,
   the video_viewer() and game_update() functions are also triggered. The global variable 'game_running' is set to
   True upon login, which keeps running these two functions until it is set to False upon finishing the game.

   returns:
   return render_template('game.html')
   """
   wit_voor = request.form['wit_voor']
   wit_achter = request.form['wit_achter']
   zwart_voor = request.form['zwart_voor']
   zwart_achter = request.form['zwart_achter']

   # Gets corresponding user ID' for a selected username for a game
   def get_ID(username):
       cur.execute("SELECT ID FROM Players WHERE username = %s", (username,))
       ID = cur.fetchone()[0]
       return int(ID)

   # Warn user if not all required names are selected
   if wit_voor == 'Naam' or wit_achter == 'Naam' or zwart_achter == 'Naam' or zwart_voor == 'Naam':
       flash('Selecteer voor iedere positie een speler')
       return redirect(url_for('index'))

   else:
       # Get ID's corresponding to the selected usernames
       global witvoor_ID
       global witachter_ID
       global zwartvoor_ID
       global zwartachter_ID
       witvoor_ID = get_ID(wit_voor)
       witachter_ID = get_ID(wit_achter)
       zwartvoor_ID = get_ID(zwart_voor)
       zwartachter_ID = get_ID(zwart_achter)

       # Initialize timer, game, and game_running variable
       global start_time
       global game
       global game_running

       game = game_data()
       game.start()
       start_time = game.time
       game_running = True

       return render_template('game.html')

def video_stream():
    """
    Responsible for streaming live video from webcam to game.html. The function is wrapped inside of
    app.response_class which actually enables the live-streaming of the video. All tracking and updating of
    the global game_data() object also happens within this function. After a game is finished,
    relevant game data is written to the MySQL database and relevant global variables are set to None so
    the next game can be started.
    """
    global video_camera
    global score
    global start_time
    global game_running

    # Start videocapture
    if video_camera == None:
        video_camera = cv2.VideoCapture(path.dirname(path.dirname(path.abspath(__file__))) + "\\data\\game.mp4")

    # Keep running until game_running == False
    while game_running:
        global game
        ok, frame = video_camera.read()

        # Mock scoring update
        score_black, score_white = np.random.choice([0, 1], 1, p=[0.95, 0.05]), np.random.choice([0, 1], 1, p=[0.95, 0.05])
        score[0] += score_black[0]
        score[1] += score_white[0]

        if score[0] == 5 or score[1] == 5:

            # Stop game and yield last frame
            game.stop()
            game_running = False
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + global_frame + b'\r\n\r\n')

            # Convert starttime to right format
            db_starttime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))

            # Write game data to MySQL database (Table: Games)
            cur.execute("INSERT INTO Games (PlayerID_Black1, PlayerID_Black2, PlayerID_White1, PlayerID_White2,"
                        "StartTime, Duration, ScoreWhite, ScoreBlack) VALUES ({}, {}, {}, {}, '{}', '{}', {}, {})".format(
                        zwartachter_ID, zwartvoor_ID, witachter_ID, witvoor_ID, db_starttime, game.duration, score[1], score[0]))
            db.commit()

            # Fetch current game_ID
            cur.execute("SELECT ID FROM Games ORDER BY ID DESC LIMIT 1")
            game_id = cur.fetchone()[0]

            prev = 0
            for field_index, field in enumerate(game.fields):
                hull = field.hull_to_string()
                cur.execute("INSERT INTO Hulls (Hull) VALUES ('{}')".format(hull))
                hullid = cur.lastrowid

                for frame_no, dp in enumerate(game.datapoints[prev:]):
                    if dp.field_index == field_index:
                        x_pos = dp.pos[0]
                        y_pos = dp.pos[1]
                        speed = dp.speed
                        angle = dp.angle
                        accuracy = dp.accuracy

                        if accuracy == 1:
                            accuracy = 0.999

                        if dp.hit:
                            cur.execute(
                                "INSERT INTO Datapoints (FrameNumber, GameID, HullID, XCoord, YCoord, Speed, Angle, Accuracy,"
                                "HitType) VALUES ({}, {}, {}, {}, {}, {}, {}, {}, {})".format(
                                    frame_no, game_id, hullid, x_pos, y_pos, speed, angle, accuracy, dp.hit.to_int()))
                        else:
                            cur.execute(
                                "INSERT INTO Datapoints (FrameNumber, GameID, HullID, XCoord, YCoord, Speed, Angle, Accuracy)"
                                "VALUES ({}, {}, {}, {}, {}, {}, {}, {})".format(
                                    frame_no, game_id, hullid, x_pos, y_pos, speed, angle, accuracy))

                        db.commit()

                    else:
                        break

                prev = frame_no

            # Set relevant globals to None for next game
            video_camera = None
            print("Game finished!")

        # If the game is not finished, keep tracking and streaming frames to webpage
        if ok:
            _, jpeg = cv2.imencode('.jpg', frame)
            global_frame = jpeg.tobytes()
            jpeg = jpeg.tobytes()
            track(frame, game, 1)

            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n\r\n')

@app.route('/video_viewer')
def video_viewer():
    return app.response_class(video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def update_dashboard():
    """
    Updates the scores and timer in game.html. The function is wrapped inside of
    app.response_class().
    """
    global start_time
    global score
    one_more = True

    # Two while loops have to be used to display right score on scoreboard after game completion
    while one_more:
        while game_running:

            # Calculate current time and yield to webpage
            seconds = time.time() - start_time
            m, s = divmod(seconds, 60)
            h, m = divmod(m, 60)

            yield('{:.0f}m{:.0f}s {} {}\n'.format(m, s, score[0], score[1]))

        # One more loop after game_running has turned to False
        yield ('{:.0f}m{:.0f}s {} {}\n'.format(m, s, score[0], score[1]))
        one_more = False

@app.route('/game_update')
def game_update():
    return app.response_class(update_dashboard(),
                              mimetype='text/plain')

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
