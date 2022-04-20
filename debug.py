
# IGNORE THIS FILE, JUST FOR DEBUGGING
from redherring import make

env = make(
    domain_name="cheetah",   # env name, for example "cheetah"
    task_name="run",
    resource_files="distractors/train/*.mp4", # resource_files == disctractors??
    img_source='video',#'video', # type of resource_files?
    total_frames=1000,  # number of frames to get from image before it loops? idk
    seed=8,
    visualize_reward=False, # what is this
    from_pixels=True, # what type of obs
    height=84, # image obs height
    width=84, # image obs width
    frame_skip=1   # skip frames like is done for atari envs
)
env.step(env.action_space.sample())
env.render()
