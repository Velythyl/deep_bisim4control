# Distractor Environments

A series of DeepMind Control environments with distractor backgrounds to test resistance to distractors.

## Instructions

```shell
python3 -m venv venv
source venv/bin/activate

pip3 install -r requirements.txt
```

## Running the natural video setting

You can download the Kinetics 400 dataset and grab the driving_car label from the train dataset. This is the dataset used
by [the original paper](https://github.com/facebookresearch/deep_bisim4control). Of course, you may use any other video dataset.

Some instructions for downloading the dataset can be found here: https://github.com/Showmax/kinetics-downloader. Just download
it and drag-and-drop it to [distractors](redherring/distractors).

## License

This project is licensed under the CC-BY-NC 4.0, as found in the [LICENSE](./LICENSE) file of this repo.

## Attribution Notice

Original code by FacebookResearch for the "Learning Invariant Representations for Reinforcement Learning without Reconstruction" paper.

Original repo is [here](https://github.com/facebookresearch/deep_bisim4control).

Formal attribution: "Learning Invariant Representations for Reinforcement Learning without Reconstruction" by [https://github.com/facebookresearch/deep_bisim4control](https://github.com/facebookresearch/deep_bisim4control) is licensed under CC-BY-NC 4.0. 
