import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
from utils import read_video, save_video
from trackers import Tracker
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
import subprocess

def process_video(input_path, output_path):
    video_frames = read_video(input_path)
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')
    tracker.add_position_to_tracks(tracks)

    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames, read_from_stub=True, stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else -1)
    team_ball_control = np.array(team_ball_control)

    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    save_video(output_video_frames, output_path)

def extract_video_preview(video_path):
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()
    if success:
        preview_path = video_path + "_preview.jpg"
        cv2.imwrite(preview_path, frame)
        return preview_path
    return None

def update_output_list(list_frame):
    for widget in list_frame.winfo_children():
        widget.destroy()

    output_folder = 'output_videos'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(output_folder):
        if filename.endswith(('.mp4', '.avi')):
            video_path = os.path.join(output_folder, filename)
            preview_path = extract_video_preview(video_path)

            frame = tk.Frame(list_frame, bg="#f8f9fa", highlightbackground="#ced4da", highlightthickness=1)
            frame.pack(fill=tk.X, pady=5, padx=10)
            frame.bind("<Enter>", lambda e, f=frame: f.config(bg="#e9ecef"))
            frame.bind("<Leave>", lambda e, f=frame: f.config(bg="#f8f9fa"))

            if preview_path and os.path.exists(preview_path):
                img = Image.open(preview_path)
                img.thumbnail((80, 80))
                img = ImageTk.PhotoImage(img)
                preview_label = tk.Label(frame, image=img, bg="#f8f9fa")
                preview_label.image = img
                preview_label.pack(side=tk.LEFT, padx=5)

            label = tk.Label(frame, text=filename, anchor="w", bg="#f8f9fa", font=("Arial", 12))
            label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

            play_button = tk.Button(
                frame, text="Play", bg="#007bff", fg="white", font=("Arial", 10),
                command=lambda p=video_path: play_video(p)
            )
            play_button.pack(side=tk.RIGHT, padx=5)

            delete_button = tk.Button(
                frame, text="Delete", bg="#dc3545", fg="white", font=("Arial", 10),
                command=lambda p=video_path: delete_video(p, list_frame)
            )
            delete_button.pack(side=tk.RIGHT, padx=5)


def play_video(video_path):
    try:
        subprocess.run(['start', video_path], shell=True)
    except Exception as e:
        messagebox.showerror("Error", f"Could not play video: {e}")

def delete_video(video_path, list_frame):
    try:
        if os.path.exists(video_path):
            os.remove(video_path)

        preview_path = video_path + "_preview.jpg"
        if os.path.exists(preview_path):
            os.remove(preview_path)

        update_output_list(list_frame)
    except Exception as e:
        messagebox.showerror("Error", f"Could not delete video: {e}")

def select_and_process_file(list_frame):
    input_folder = 'input_videos'
    output_folder = 'output_videos'

    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi")])
    if file_path:
        input_filename = os.path.basename(file_path)
        input_path = os.path.join(input_folder, input_filename)
        output_path = os.path.join(output_folder, f"output_{input_filename}")

        with open(file_path, 'rb') as src, open(input_path, 'wb') as dst:
            dst.write(src.read())

        try:
            process_video(input_path, output_path)
            messagebox.showinfo("Success", f"Video processed and saved as {output_path}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while processing the video: {e}")

        update_output_list(list_frame)

def main():
    root = tk.Tk()
    root.title("Football Analysis System")
    root.geometry("1000x700")

    style = ttk.Style()
    style.theme_use("clam")

    output_label = ttk.Label(root, text="Output Videos:", font=("Arial", 16))
    output_label.pack(pady=10)

    list_canvas = tk.Canvas(root)
    list_frame = tk.Frame(list_canvas)

    scrollbar = ttk.Scrollbar(root, orient=tk.VERTICAL, command=list_canvas.yview)
    list_canvas.configure(yscrollcommand=scrollbar.set)

    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    list_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    list_canvas.create_window((0, 0), window=list_frame, anchor="nw")

    list_frame.bind("<Configure>", lambda e: list_canvas.configure(scrollregion=list_canvas.bbox("all")))

    update_output_list(list_frame)

    process_button = ttk.Button(root, text="Select and Process Video", command=lambda: select_and_process_file(list_frame))
    process_button.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()
