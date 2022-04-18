from maker import make

# IGGNORE THIS FILE, JUST FOR DEBUGGING

env = make(
    domain_name="cheetah",   # env name, for example "cheetah"
    task_name="run",
    resource_files="distractors/train/*.mp4", # resource_files == disctractors??
    img_source='noise',#'video', # type of resource_files?
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

"""
parser.add_argument('--domain_name', default='cheetah')
parser.add_argument('--task_name', default='run')
parser.add_argument('--image_size', default=84, type=int)
parser.add_argument('--action_repeat', default=1, type=int)
parser.add_argument('--frame_stack', default=3, type=int)
parser.add_argument('--resource_files', type=str)
parser.add_argument('--eval_resource_files', type=str)
parser.add_argument('--img_source', default=None, type=str, choices=['color', 'noise', 'images', 'video', 'none'])
parser.add_argument('--total_frames', default=1000, type=int)
"""