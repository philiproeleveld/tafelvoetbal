# Tafelvoetbal Tracking

The goal of this project is to automate the arbitration of foosball given the (live) video footage of a game.

### Documentation

`track.py` contains the functional tracking code. Before running this script, be sure to install the requirements as specified by `python_requirements.txt`. The script can be executed with a multitude of optional arguments. The script can be called in two ways. The first way is simply to run the script directly. The optional arguments can be passed as flags. See below for a list of these flags.

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
    Specify a path to the video source. (Default: unspecified, uses the webcam instead)

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

- `--wait-on-hits N`  
    Specify how long at least to wait on frames where a hit occurs in milliseconds. (Default: equal to the value specified for `--wait`)

- `--scaledown N`  
    Specify how much to scale down footage before doing calculations. Recommended for footage of more than 1 megapixel. (Default: 1, i.e. no scaledown)

### Footage

Some sample footage is included in this repository. The video file `game.mp4` is an excerpt of a game. The footage contains some gameplay and ends shortly after a single goal is made. Since the footage is quite long, `game_short.mp4` is included as well. This video contains only the last twenty seconds of `game.mp4` in which the goal is made. However, `game_short.mp4` has horizontally flipped the footage of `game.mp4`, so be sure to use the `--flip` flag to flip it back if desired.
