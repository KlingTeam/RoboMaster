import gradio as gr
import os
from gradio_image_prompter import ImagePrompter
import numpy as np
import decord
import cv2
from PIL import Image, ImageDraw
from io import BytesIO
import imageio.v2 as imageio
from gradio_image_prompter_user import ImagePrompter

def process_points(points, frames):

    if len(points) >= frames:

        frames_interval = np.linspace(0, len(points) - 1, frames, dtype=int)
        points = [points[i] for i in frames_interval]
        return points

    else:

        insert_num = frames - len(points)
        insert_num_dict = {}
        interval = len(points) - 1
        n = insert_num // interval
        for i in range(interval):
            insert_num_dict[i] = n

        m = insert_num % interval
        if m > 0:
            frames_interval = np.linspace(0, len(points)-1, m, dtype=int)
            if frames_interval[-1] > 0:
                frames_interval[-1] -= 1
            for i in range(interval):
                if i in frames_interval:
                    insert_num_dict[i] += 1

        res = []
        for i in range(interval):
            insert_points = []
            x0, y0 = points[i]
            x1, y1 = points[i + 1]

            delta_x = x1 - x0
            delta_y = y1 - y0

            for j in range(insert_num_dict[i]):
                x = x0 + (j + 1) / (insert_num_dict[i] + 1) * delta_x
                y = y0 + (j + 1) / (insert_num_dict[i] + 1) * delta_y
                insert_points.append([int(x), int(y)])

            res += points[i : i + 1] + insert_points
        res += points[-1:]
        
        return res
    
def draw_circle(rgb, coord, radius, color=(255, 0, 0), visible=True, color_alpha=None):
    # Create a draw object
    draw = ImageDraw.Draw(rgb)
    # Calculate the bounding box of the circle
    left_up_point = (coord[0] - radius, coord[1] - radius)
    right_down_point = (coord[0] + radius, coord[1] + radius)
    # Draw the circle
    color = tuple(list(color) + [color_alpha if color_alpha is not None else 255])

    draw.ellipse(
        [left_up_point, right_down_point],
        fill=tuple(color) if visible else None,
        outline=tuple(color),
    )
    return rgb

def save_images2video(images, video_name, fps):
    fps = fps
    format = "mp4"
    codec = "libx264" 
    ffmpeg_params = ["-crf", str(12)]
    pixelformat = "yuv420p" 
    video_stream = BytesIO()

    with imageio.get_writer(
        video_stream,
        fps=fps,
        format=format,
        codec=codec,
        ffmpeg_params=ffmpeg_params,
        pixelformat=pixelformat,
    ) as writer:
        for idx in range(len(images)):
            writer.append_data(images[idx])
    
    video_data = video_stream.getvalue()
    output_path = os.path.join(video_name + ".mp4")
    with open(output_path, "wb") as f:
        f.write(video_data)
    
def process_obj(prompts, sample_name, start_frame, end_frame):

    total_frames = 37
    image_path = f"output/{sample_name}.png"
    img = cv2.imread(image_path)[:,:,::-1]
    video_chunk = np.repeat(img[None], total_frames, axis=0)
    color = np.array([0,255,0])
    
    start_frame = int(start_frame)
    end_frame = int(end_frame)
    action_frames = end_frame - start_frame + 1
    track = np.load('coord.npy').tolist()
    track = process_points(track, action_frames)

    track = [track[0]] * start_frame + track + [track[-1]] * (total_frames - end_frame - 1)
    assert len(track) == total_frames
    track = np.array(track)
    np.save('coord_extend.npy', track.astype(np.int32))

    for t in range(total_frames):
        img = Image.fromarray(np.uint8(video_chunk[t]))
        coord = (track[t, 0], track[t, 1])
        visibile = True
        if coord[0] != 0 and coord[1] != 0:
            img = draw_circle(
                img,
                coord=coord,
                radius=12,
                color=color,
                visible=visibile,
                color_alpha=255,
            )
        video_chunk[t] = np.array(img)
    
    cv2.imwrite('intermediate.png', video_chunk[0,:,:,::-1])
    save_images2video(video_chunk, 'track_blended', total_frames//3)

    # return prompts["points"], 'track_blended.mp4', 'intermediate.png'
    return 'track_blended.mp4', 'intermediate.png'


def process_robot_pre(prompts, sample_name, start_frame, end_frame):

    video_path = f"output/{sample_name}.mp4"
    cap = cv2.VideoCapture(os.path.join(video_path))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ctx = decord.cpu(0)
    reader = decord.VideoReader(os.path.join(video_path), ctx=ctx, height=height, width=width)
    frame_indexes = [frame_idx for frame_idx in range(len(reader))]
    try:
        video_chunk = reader.get_batch(frame_indexes).asnumpy()    
    except:
        video_chunk = reader.get_batch(frame_indexes).numpy()
    color = np.array([255,0,0])
    
    action_frames = int(start_frame) 
    track = np.load('coord.npy').tolist()
    track = np.array(process_points(track, action_frames))
    track_obj = np.load(f'output/{sample_name}_obj.npy')
    track_obj = track_obj[int(start_frame):int(end_frame)+1]

    track = np.concatenate((track, track_obj - track_obj[0] + track[-1]), axis=0)
    np.save('coord_extend.npy', track.astype(np.int32))

    for t in range(total_frames):
        img = Image.fromarray(np.uint8(video_chunk[t]))
        if t <= action_frames:
            coord = (track[t, 0], track[t, 1])
        else:
            coord = (track[action_frames-1, 0], track[action_frames-1, 1])
        visibile = True
        if coord[0] != 0 and coord[1] != 0:
            img = draw_circle(
                img,
                coord=coord,
                radius=12,
                color=color,
                visible=visibile,
                color_alpha=255,
            )

        video_chunk[t] = np.array(img)
    
    cv2.imwrite('intermediate.png', video_chunk[t,:,:,::-1])
    save_images2video(video_chunk, 'track_blended', total_frames//3)

    # return prompts["points"], 'track_blended.mp4', 'intermediate.png'
    return 'track_blended.mp4', 'intermediate.png'

def process_robot_post(prompts, sample_name, start_frame, end_frame):

    video_path = f"output/{sample_name}.mp4"
    cap = cv2.VideoCapture(os.path.join(video_path))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ctx = decord.cpu(0)
    reader = decord.VideoReader(os.path.join(video_path), ctx=ctx, height=height, width=width)
    frame_indexes = [frame_idx for frame_idx in range(len(reader))]
    try:
        video_chunk = reader.get_batch(frame_indexes).asnumpy()    
    except:
        video_chunk = reader.get_batch(frame_indexes).numpy()
    color = np.array([255,0,0])

    track_robot = np.load(f'output/{sample_name}_robot.npy')
    action_frames = total_frames - int(end_frame) - 1
    track = np.load('coord.npy').tolist()
    track = np.array(process_points(track, action_frames))
    track = np.concatenate((track_robot, track), axis=0)
    assert len(track) == total_frames
    np.save('coord_extend.npy', track.astype(np.int32))

    for t in range(total_frames):
        if t >= len(track_robot):
            img = Image.fromarray(np.uint8(video_chunk[t]))
            coord = (track[t, 0], track[t, 1])
            visibile = True
            if coord[0] != 0 and coord[1] != 0:
                img = draw_circle(
                    img,
                    coord=coord,
                    radius=12,
                    color=color,
                    visible=visibile,
                    color_alpha=255,
                )
            video_chunk[t] = np.array(img)
    
    cv2.imwrite('intermediate.png', video_chunk[t,:,:,::-1])
    save_images2video(video_chunk, 'track_blended', total_frames//3)

    # return prompts["points"], 'track_blended.mp4', 'intermediate.png'
    return 'track_blended.mp4', 'intermediate.png'

def save_prompt_fn(sample_name, text_prompt):
    with open(f'output/{sample_name}.txt', 'w') as f:
        f.write(f'{text_prompt}')
    return f'prompt of sample {sample_name} is saved'

def save_track_robot_pre_fn(sample_name):
    os.system(f'rm -rf intermediate.png')
    os.system(f'rm -rf coord.npy')
    os.system(f'mv coord_extend.npy output/{sample_name}_robot.npy')
    os.system(f'mv track_blended.mp4 output/{sample_name}.mp4')
    return f'robot pre-track of sample {sample_name} is done'

def save_track_robot_post_fn(sample_name):
    os.system(f'rm -rf intermediate.png')
    os.system(f'rm -rf coord.npy')
    os.system(f'mv coord_extend.npy output/{sample_name}_robot.npy')
    os.system(f'mv track_blended.mp4 output/{sample_name}.mp4')
    return f'robot post-track of sample {sample_name} is done'

def save_track_obj_fn(sample_name, start_frame, end_frame):
    transit = np.array([int(start_frame), int(end_frame)])
    np.save(f'output/{sample_name}_transit.npy', transit)
    os.system(f'rm -rf intermediate.png')
    os.system(f'rm -rf coord.npy')
    os.system(f'mv coord_extend.npy output/{sample_name}_obj.npy')
    os.system(f'mv track_blended.mp4 output/{sample_name}.mp4')
    return f'obj track of sample {sample_name} is done'

def save_mask_fn(sample_name, mask_image):
    mask = mask_image['layers'][0][:,:,3] == 255
    np.save('output/'+sample_name+'_obj_mask.npy', mask)
    return f'obj mask of sample {sample_name} is done'

with gr.Blocks() as demo:

    gr.Markdown("## Input 1: Sample Name & Prompt")
    with gr.Row():
        sample_name = gr.Textbox(label="Sample Name")
        text_prompt = gr.Textbox(label="Please Input Prompt")
    save_prompt = gr.Button(value="Save Prompt", visible=True)

    with gr.Row():
        with gr.Column():
            gr.Markdown("## Input 2: Object Mask")
            mask_image = gr.ImageEditor(label="Image Mask", height=600)

        with gr.Column():
            gr.Markdown("## Input 3: Collaborative Trajectory")
            input_image = ImagePrompter(show_label="Input Image")
        
    with gr.Row():

        with gr.Column():
            gr.Markdown("## Output 1: Intermediate Image")
            intermediate_image = gr.Image(label="Intermediate Image")

        with gr.Column():
            # gr.Markdown("## Output 2: Trajectory (Point Sets)")
            # points_output = gr.Dataframe(label="Points")

            gr.Markdown("## Output 2: Composite Video")
            output_video = gr.Video(label="Output Video", height=480)

    with gr.Row():

        with gr.Column():
            
            gr.Markdown("## Step 1: Input Object Mask")
            save_mask = gr.Button("Save Mask")
        
            gr.Markdown("## Step 2: Input Object Tracks")
            with gr.Row():
                start_frame = gr.Textbox(label="Start Frame (>=0)")
                end_frame = gr.Textbox(label="End Frame (<=36)")
            with gr.Row():
                submit_obj_button = gr.Button(value="Submit & Load", visible=True)
                save_obj_button = gr.Button(value="Save", visible=True)

        with gr.Column():

            gr.Markdown("## Step 3: Input Robot Arm Tracks")

            gr.Markdown("#### Pre-Interaction")
            with gr.Row():
                submit_robot_pre_button = gr.Button(value="Submit & Load", visible=True)
                save_robot_pre_button = gr.Button(value="Save", visible=True)
            
            gr.Markdown("#### Post-Interaction")
            with gr.Row():
                submit_robot_post_button = gr.Button(value="Submit & Load", visible=True)
                save_robot_post_button = gr.Button(value="Save", visible=True)

            text_output = gr.Textbox(label="Status", show_label=True)
    
    save_prompt.click(fn=save_prompt_fn, inputs=[sample_name, text_prompt], outputs=[text_output])
    save_mask.click(fn=save_mask_fn, inputs=[sample_name, mask_image], outputs=text_output)
    submit_obj_button.click(fn=process_obj, inputs=[input_image, sample_name, start_frame, end_frame], outputs=[output_video,intermediate_image])
    save_obj_button.click(fn=save_track_obj_fn, inputs=[sample_name, start_frame, end_frame], outputs=[text_output])    
    submit_robot_pre_button.click(fn=process_robot_pre, inputs=[input_image, sample_name, start_frame, end_frame], outputs=[output_video, intermediate_image])
    save_robot_pre_button.click(fn=save_track_robot_pre_fn, inputs=[sample_name], outputs=[text_output])
    submit_robot_post_button.click(fn=process_robot_post, inputs=[input_image, sample_name, start_frame, end_frame], outputs=[output_video, intermediate_image])
    save_robot_post_button.click(fn=save_track_robot_post_fn, inputs=[sample_name], outputs=[text_output])

demo.launch()
