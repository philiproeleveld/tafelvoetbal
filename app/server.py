from flask import Flask, render_template, redirect, request, flash, url_for
import cv2
import MySQLdb
import time
from os import sys, path

repo = path.dirname(path.dirname(path.abspath(__file__)))
if repo.find('/') == -1:
    track_dir = repo + "\\track"
    video_file = repo + "\\data\\game.mp4"
else:
    track_dir = repo + "/track"
    video_file = repo + "/data/game2.mov"

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))) + "/track")

from track import track
from game_data import game_data

app = Flask(__name__)
app.secret_key = 'cookie'

# Setup database connection
db = MySQLdb.connect(host="localhost",
                     user="root",
                     passwd="password",
                     db="Tafelvoetbal")

cur = db.cursor()

# GAME GLOBALS (set to None or overwritten at start of each new game)
wit_voor = None         # Username player in the front (white)
wit_achter = None       # Username player in the back (white)
zwart_voor = None       # Username player in the front (black)
zwart_achter = None     # Username player in the back (black)
witvoor_ID = None       # ID player in the front (white)
witachter_ID = None     # ID player in the back (white)
zwartvoor_ID = None     # ID player in the front (black)
zwartachter_ID = None   # ID player in the back (black)
video_camera = None     # Used for storing cv2 VideoCapture object
last_scored = None      # Keeps track of last scoring team, as to disqualify goal if player chooses so
game = None             # Contains the game data
game_running = False    # True if there is currently a game being played
m = None                # m stores minutes of playing time
s = None                # s stores seconds of playing time

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

# Redirects to login after a game is finished (requested client-side in game.html)
@app.route('/', methods=['POST'])
def login_redirect():
    # Login page includes all current usernames in database
    cur.execute("SELECT Username FROM Players")
    all_usernames = [row[0] for row in cur.fetchall()]

    return render_template('index.html', db_names=all_usernames)

@app.route('/handle_register', methods=['POST'])
def handle_register():
    """
    Handles registration of users when the register button is clicked in register.html.
    It flashes an error on the registration page if the requirements are not met, and
    writes the registration to the MySQL database (table: Players) when the requirements for registration
    are met, after which the login-page template is rendered.

    returns:
    redirect(url_for('register'))                           if requirements for registration are not met
    render_template('index.html', db_names=all_usernames)   if requirements for registration are met
    """

    # Acquire form variables from register.html
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
    elif ' ' in username:
        error = "Username mag geen spaties bevatten"
    elif age.isdigit() == False:
        error = "Leeftijd dient een getal te zijn"
    elif int(age) > 70:
        error = 'Leeftijd mag niet hoger zijn dan 70'
    elif int(age) < 18:
        error = 'Leeftijd mag niet lager zijn dan 18'

    # Flash message on register page to user if register form is not filled in correctly
    if 'error' in locals():
        flash(error)
        return redirect(url_for('register'))

    # Write register data to SQL database and return to the login page when registration is legitimate
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

        # Retrieve username list to pass to login page for filling selector buttons
        query = cur.execute("SELECT Username FROM Players")
        all_usernames = [row[0] for row in cur.fetchall()]

        return render_template('index.html', db_names=all_usernames)

@app.route('/game', methods=['POST'])
def handle_login():
   """
   Handles the start of a game when login button is clicked in index.html. Upon start of the game
   the user ID's that match to the selected usernames are retrieved from the MySQL database (table: Players) and
   are saved into global variables. After this, game.html is rendered. When this template is rendered,
   the update_dashboard() function is also triggered. Besides, the global variable 'game_running' is set to
   True upon, which assures that the updating function keeps running until it is set to False upon
   finishing the game.

   returns:
   return render_template('game.html')
   """

   # Form variables
   wit_voor = request.form['wit_voor']
   wit_achter = request.form['wit_achter']
   zwart_voor = request.form['zwart_voor']
   zwart_achter = request.form['zwart_achter']

   # Gets corresponding user ID for a selected username
   def get_ID(username):
       cur.execute("SELECT ID FROM Players WHERE username = %s", (username,))
       ID = cur.fetchone()[0]
       return int(ID)

   # Warn user if the required fields are not filled
   if wit_voor == 'Naam' or wit_achter == 'Naam' or zwart_achter == 'Naam' or zwart_voor == 'Naam':
       flash('Selecteer voor iedere positie een speler')
       return redirect(url_for('index'))

   else:
       # Get player ID's corresponding to the selected usernames
       global witvoor_ID
       global witachter_ID
       global zwartvoor_ID
       global zwartachter_ID
       witvoor_ID = get_ID(wit_voor)
       witachter_ID = get_ID(wit_achter)
       zwartvoor_ID = get_ID(zwart_voor)
       zwartachter_ID = get_ID(zwart_achter)

       # Initialize timer, game, and game_running
       global game
       global game_running

       game = game_data()
       game.start()

       game_running = True

       return render_template('game.html')

@app.route('/adjust_score', methods=['POST'])
def adjust_score():
    '''
    Responsible for adjusting score when player requests to do so in game.html. If the increase
    variable passed from the POST request equals 'increase_white' or 'increase_black', the relevant score
    is increased. Otherwise, the team that last scored gets a decrease of 1 on its score.

    returns:
    return redirect(url_for('login'))
    '''

    global game
    global last_scored
    global game_running

    increase = request.form['team_to_adjust']

    if increase:

        # Find the last hit made by team specified in 'increase' in datapoints list and update score/hit
        for idx, dp in reversed(list(enumerate(game.datapoints))):
            if dp.hit is None:
                continue
            else:
                if increase == 'increase_white':
                    if dp.hit.team == 1: # White hit
                        game.datapoints[idx].hit.goal = 0
                        game.score[1] += 1
                        break

                elif increase == 'increase_black':
                    if dp.hit.team == 0: # Black hit
                        game.datapoints[idx].hit.goal = 1
                        game.score[0] += 1
                        break

    else:
        if last_scored == 'white':
            game.score[1] -= 1
        elif last_scored == 'black':
            game.score[0] -= 1

        # Find the last hit in datapoints list and set self.goal to None for this datapoint
        for idx, dp in reversed(list(enumerate(game.datapoints))):
            if dp.hit is None:
                continue
            else:
                if dp.hit.goal:
                    game.datapoints[idx].goal = None
                    break

    return redirect(url_for('login'))

def update_dashboard():
    """
    Updates the scores and timer in game.html for the duration of one game. The function is wrapped inside of
    app.response_class(). Within the function, new video-frames are continuously generated from a live-feed
    which are processed in the track() function. After requirements are met for finishing a game, the game is stopped
    and relevant data is written to the MySQL database. After this, a new game can be started.
    """

    global video_camera
    global game_running
    global last_scored
    global game

    one_more = True     # Used to show last frame
    prev_black = 0      # Keeps track of last score black
    prev_white = 0      # Keeps track of last score white

    # Start videocapture
    if video_camera == None:
        # video_camera = cv2.VideoCapture(video_file)
        video_camera = cv2.VideoCapture(1)

    # Keep running until game_running == False
    while one_more:
        time.sleep(1)
        while game_running:
            ok, frame = video_camera.read()

            # Mock scoring update, uncomment when using video file for testing purposes
            # score_black, score_white = np.random.choice([0, 1], 1, p=[0.99, 0.01]), np.random.choice([0, 1], 1, p=[0.99, 0.01])
            # game.score[0] += score_black[0]
            # game.score[1] += score_white[0]

            # Determine the last scoring team (for score adjustment purposes)
            if game.score[1] > prev_white:
                prev_white = game.score[1]
                last_scored = 'white'

            elif game.score[0] > prev_black:
                prev_black = game.score[0]
                last_scored = 'black'

            # Stop game if there is a score of 10 or higher and there is a difference of 2
            if game.score[0] >= 10 or game.score[1] >= 10:
                if abs(game.score[0] - game.score[1]) >= 2:

                    # Stop game and yield last frame
                    game.stop()
                    game_running = False

                    # Write game data to MySQL database
                    game.write_db(zwartachter_ID, zwartvoor_ID, witachter_ID, witvoor_ID)

                    # Set relevant globals to None for next game
                    video_camera = None

            # If the game is not finished, keep tracking and streaming frames to webpage
            if ok:
                track(frame, game, 2)

                # Globals to update m and s for last frame
                global m
                global s

                # Calculate current time and yield to webpage
                seconds = time.time() - game.time
                m, s = divmod(seconds, 60)
                h, m = divmod(m, 60)

                yield ('{:.0f}m{:.0f}s {} {}\n'.format(m, s, game.score[1], game.score[0]))

        # One more loop after game_running has turned to False to avoid client-side empty scoreboard
        yield ('{:.0f}m{:.0f}s {} {}\n'.format(m, s, game.score[1], game.score[0]))
        one_more = False

@app.route('/game_update')
def game_update():
    return app.response_class(update_dashboard(),
                              mimetype='text/plain')

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
