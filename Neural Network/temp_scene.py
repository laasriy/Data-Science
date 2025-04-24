from manimlib import *

class NeuronAnimation(Scene):
    def construct(self):
        circle = Circle(color=BLUE, fill_opacity=0.5)
        self.play(Create(circle))
        self.wait(1)

# 2. Exécuter avec le chemin absolu
current_dir = os.getcwd()
manim_path = os.path.join(current_dir, "manim", "manim.py")  # Chemin correct

subprocess.run([
    "python",
    manim_path,
    "temp_scene.py",
    "NeuronAnimation",
    "-o", "animation.mp4",
    "--media_dir", current_dir,
    "--use_opengl_renderer",
    "--quiet"
])

# 3. Afficher le résultat
Video("media/videos/NeuronAnimation/480p15/animation.mp4", width=600)
