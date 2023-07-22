# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Base, Model Input, Path

from inference import convert_video

class Output(BaseModel):
    composition: Path
    alpha: Path
    foreground: Path

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = torch.load("./rvm_mobilenetv3.pth")

    def predict(
        self,
        video: Path = Input(description="Input Video"),
    ) -> Path:
        """Run a single prediction on the model"""
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)
        convert_video(
            model,                           # The loaded model, can be on any device (cpu or cuda).
            input_source=video,        # A video file or an image sequence directory.
            output_type='video',             # Choose "video" or "png_sequence"
            output_composition='com.mp4',    # File path if video; directory path if png sequence.
            output_alpha="pha.mp4",          # [Optional] Output the raw alpha prediction.
            output_foreground="fgr.mp4",     # [Optional] Output the raw foreground prediction.
            output_video_mbps=4,             # Output video mbps. Not needed for png sequence.
            seq_chunk=12,                    # Process n frames at once for better parallelism.
            num_workers=1,                   # Only for image sequence input. Reader threads.
            progress=True                    # Print conversion progress.
        )

        return Output(composition='com.mp4', alpha='pha.mp4', foreground='fgr.mp4')
