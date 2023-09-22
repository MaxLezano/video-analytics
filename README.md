# Renard Video Analytics

## How to install

To install the necessary dependencies we use PIPENV. You can install it using the following command:


1. Install the dependencies:

   ```bash
   pipenv install
   ```

## How to run

To run one of the applications (possession computation and passes counter) you need to use flags in the console.

These flags are defined in the following table:

| Argument | Description | Default value |
| ----------- | ----------- | ----------- |
| application | Set it to `possession` to run the possession counter or `passes` if you like to run the passes counter | None, but mandatory |
| path-to-the-model | Path to the soccer ball model weights (`pt` format) | `/models/ball.pt` |
| path-to-the-video | Path to the input video | `/videos/soccer_possession.mp4` |

The following command shows you how to run this project.

```
python run.py --<application> --model <path-to-the-model> --video <path-to-the-video>
```

>__Warning__: You have to run this command on the root of the project folder.

Here is an example on how to run the command:
    
```bash
python main.py --possession --model models/ball.pt --video videos/soccer_possession.mp4
```

An mp4 video will be generated after the execution. The name is the same as the input video with the suffix `_out` added.
