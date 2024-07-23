# Reference code: https://github.com/li-plus/DSNet/blob/1804176e2e8b57846beb063667448982273fca89/src/make_dataset.py#L4
# Reference code: https://github.com/e-apostolidis/PGL-SUM/blob/81d0d6d0ee0470775ad759087deebbce1ceffec3/model/configs.py#L10
import cv2
import torch

from pathlib import Path
from tqdm import tqdm

from config import get_config
from generate_summary import generate_summary
from model import set_model
from video_helper import VideoPreprocessor

def pick_frames(video_path, selections):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    n_frames = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break
        
        if selections[n_frames]:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        n_frames += 1
    cap.release()

    return frames

def produce_video(save_path, frames, fps, frame_size):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, frame_size)
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.write(frame)
    out.release()

def main():
    # Load config
    config = get_config()

    # create output directory
    out_dir = Path(config.save_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # feature extractor
    video_proc = VideoPreprocessor(
        sample_rate=config.sample_rate,
        device=config.device
    )

    # search all videos with .mp4 suffix
    if config.input_is_file:
        video_paths = [Path(config.file_path)]
    else:
        video_paths = sorted(Path(config.dir_path).glob(f'*.{config.ext}'))

    # Load CSTA weights
    model = set_model(
        model_name=config.model_name,
        Scale=config.Scale,
        Softmax_axis=config.Softmax_axis,
        Balance=config.Balance,
        Positional_encoding=config.Positional_encoding,
        Positional_encoding_shape=config.Positional_encoding_shape,
        Positional_encoding_way=config.Positional_encoding_way,
        Dropout_on=config.Dropout_on,
        Dropout_ratio=config.Dropout_ratio,
        Classifier_on=config.Classifier_on,
        CLS_on=config.CLS_on,
        CLS_mix=config.CLS_mix,
        key_value_emb=config.key_value_emb,
        Skip_connection=config.Skip_connection,
        Layernorm=config.Layernorm
    )
    model.load_state_dict(torch.load(config.weight_path, map_location='cpu'))
    model.to(config.device)
    model.eval()

    # Generate summarized videos
    with torch.no_grad():
        for video_path in tqdm(video_paths,total=len(video_paths),ncols=80,leave=False,desc="Making videos..."):
            video_name = video_path.stem
            n_frames, features, cps, pick = video_proc.run(video_path)

            inputs = features.to(config.device)
            inputs = inputs.unsqueeze(0).expand(3,-1,-1).unsqueeze(0)
            outputs = model(inputs)
            predictions = outputs.squeeze().clone().detach().cpu().numpy().tolist()
            # print(cps.shape, len(predictions), n_frames, pick.shape)
            selections = generate_summary([cps], [predictions], [n_frames], [pick])[0]

            frames = pick_frames(video_path=video_path, selections=selections)
            produce_video(
                save_path=f'{config.save_path}/{video_name}.mp4',
                frames=frames,
                fps=video_proc.fps,
                frame_size=(video_proc.frame_width,video_proc.frame_height)
            )

if __name__=='__main__':
    main()
