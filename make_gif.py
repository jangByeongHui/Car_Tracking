from moviepy.editor import *
Video_PATH=""
output_NAME=""

VideoFileClip(Video_PATH).speedx(4).write_gif(output_NAME)
