# Tafelvoetbal Tracking

The goal of this project is to automate the arbitration of foosball given the (live) video footage of a game.

### Documentation
Inside the `track` folder, `track.py` contains the functional tracking code. Before running this script, be sure to install the requirements as specified by `python_requirements.txt`.
#### track.track()
The `track.py` script contains the `track()` funciton. This function does the actual tracking, given a frame. It needs at least four arguments, as follows:

`track(frame, game, scaledown)`
- `frame`  
    A numpy array containing the video frame to do the calculations on (as given by openCV).

- `game`  
    An instance of the `game_data` class defined in `track.py`.

- `scaledown`  
    An integer specifying how much to scale down footage before doing calculations.

After tracking is done, be sure to call `game.stop()`. Example usage:
```python
import cv2
from game_data import game_data
from track import track

vid = cv2.VideoCapture("game.mp4")

game = game_data()

while True:
    ok, frame = vid.read()
    if not ok:
        break
    track(frame, game, 1)

game.stop()
```
After execution of this block of code, `game` will contain all the necessary data.

#### track.py
The `track.py` script can also be executed on its own, with a multitude of optional arguments. It can be called in two ways. The first way is simply to run the script directly. The optional arguments can be passed as flags. See below for a list of these flags.

The second way to use the script is to import it and call the main method. In this case, the optional arguments can be passed to the main function directly. The names are exactly the same as the names of flags (minus the leading `--`), except that hyphens should be replaced by underscores. As an example, the following two execute the same code:  
`python track.py -s "game.mp4" --hit-detection`  
```python
import track
track.main(source="game.mp4", hit_detection=True)
```

A complete list of optional flags is as follows:
- `-h`, `--help`  
    Show this list and exit.

- `-s SRC`, `--source SRC`  
    Specify a path to the video source. (Default: unspecified, use webcam footage instead)

- `--flip`  
    Flip the footage horizontally.

- `--ball`  
    Turn on ball tracking visualization.

- `--history`  
    Turn on recent history visualization.

- `--hits`  
    Turn on hit visualization.

- `--text`  
    Turn on textual data.

- `--field`  
    Turn on field bounding box visualization.

- `--goals`  
    Turn on goal bounding box visualization.

- `--hit-detection`  
    Turn on hit detection visualization.

- `--heatmap`  
    Show a heatmap after the video is done.

- `--wait N`  
    Specify how long at least to wait on each frame in milliseconds, before the next frame is displayed. (Default: 1 ms)

- `--scaledown N`  
    Specify how much to scale down footage before doing calculations. Recommended for footage of more than 1 megapixel. (Default: 1, i.e. no scaledown)

#### server.py
This script launches the Flask web-application. To launch the application, run `python server.py` in a terminal.

Within the app, there are three web pages:
- `index.html`
   Which is the landing page that enables players to start a game.

- `register.html`
   Which enables new players to register. Submitting the form on this page results in adding the player to the Players table and redirection to index.html.
   
- `game.html`
   Which shows a scoreboard and the total time of the game to the players.
   
When a game is started, a new `game_data` object instance is initialized. The
object is updated by using the `track()` function as described above for as long as the game runs.
After the game is finished, data in the `game_data` object is written to the MySQL database, after which the object is
deleted and the app is redirected to index.html, which allows for starting of a new game.

### Sample Footage

Some sample footage is included in this repository. It can be found in the `data` folder. The video file `game.mp4` is an excerpt of a game. The footage contains some gameplay and ends shortly after a single goal is made. Since the footage is quite long, `game_short.mp4` is included as well. This video contains only the last twenty seconds of `game.mp4` in which the goal is made. However, `game_short.mp4` has horizontally flipped the footage of `game.mp4`, so be sure to use the `--flip` flag to flip it back if desired.

