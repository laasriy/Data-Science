from manimlib import *

class SigmoidAnimation(Scene):
    def construct(self):
        axes = Axes(x_range=[-4, 4], y_range=[-1, 2])
        sigmoid = axes.get_graph(lambda x: 1 / (1 + np.exp(-x)), color=GREEN)
        self.play(ShowCreation(axes), ShowCreation(sigmoid))
        self.wait(2)

# Render and display
subprocess.run(["manimgl", "sigmoid_scene.py", "SigmoidAnimation", "-o", "sigmoid.mp4", "--media_dir", ".", "--quiet"])
Video("sigmoid.mp4", width=600)
