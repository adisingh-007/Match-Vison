from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
import cv2
import numpy as np
import os

# Global variable to store progress (in a real app, this would be stored in a database or Redis)
processing_progress = {}

def update_progress(task_id, progress, status="processing"):
    """Update the progress of a task"""
    if task_id:
        processing_progress[task_id] = {
            'progress': progress,
            'status': status
        }

def process_video(input_video_path, output_video_path, task_id=None):
    """Process a video file and generate annotated output"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

        # Reading video
        print("Reading video...")
        update_progress(task_id, 5, "reading video")
        video_frames = read_video(input_video_path)
        
        # Check if video was read successfully
        if not video_frames:
            print(f"Error: Failed to read video frames from {input_video_path}")
            update_progress(task_id, 0, "failed")
            return False
        
        print(f"Successfully read {len(video_frames)} frames from video")
        update_progress(task_id, 10, "video read")

        # Initializing Tracker
        print("Initializing tracker...")
        update_progress(task_id, 15, "initializing tracker")
        tracker = Tracker('models/best.pt')

        # Getting object tracks
        print("Getting object tracks...")
        update_progress(task_id, 20, "getting object tracks")
        tracks = tracker.get_object_tracks(video_frames,
                                        read_from_stub=False,  # Don't read from stub for new videos
                                        stub_path=None)

        # Getting object positions
        print("Adding position to tracks...")
        update_progress(task_id, 25, "adding positions")
        tracker.add_position_to_tracks(tracks)

        # Adding camera movement estimator
        print("Estimating camera movement...")
        update_progress(task_id, 30, "estimating camera movement")
        # Initializing by first frame
        camera_movement_estimator = CameraMovementEstimator(video_frames[0])
        camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=False,  # Don't read from stub for new videos
                                                                                stub_path=None)
        # Calling adjust camera position
        camera_movement_estimator.add_adjust_positions_to_tracks(
            tracks, camera_movement_per_frame)

        # Added View Transformer
        print("Transforming view...")
        update_progress(task_id, 40, "transforming view")
        view_transformer = ViewTransformer()
        view_transformer.add_transformed_position_to_tracks(tracks)

        # Interpolating/inserting ball positions
        print("Interpolating ball positions...")
        update_progress(task_id, 45, "interpolating ball positions")
        tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

        # Adding speed and distance estimator
        print("Estimating speed and distance...")
        update_progress(task_id, 50, "estimating speed and distance")
        speed_and_distance_estimator = SpeedAndDistance_Estimator()
        speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

        # Assigning player teams
        print("Assigning team colors...")
        update_progress(task_id, 60, "assigning team colors")
        team_assigner = TeamAssigner()  # initialising team assigner
        # assigning team their colours and by giving them first frame and tracks of player in first frame
        team_assigner.assign_team_color(video_frames[0],
                                        tracks['players'][0])

        # Looping over each player in each frame and assigning them to colour team
        print("Assigning players to teams...")
        update_progress(task_id, 65, "assigning players to teams")
        for frame_num, player_track in enumerate(tracks['players']):
            for player_id, track in player_track.items():
                team = team_assigner.get_player_team(video_frames[frame_num],
                                                    track['bbox'], player_id)
                tracks['players'][frame_num][player_id]['team'] = team
                tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
            
            # Update progress based on frame processing
            progress = 65 + int(20 * frame_num / len(tracks['players']))
            update_progress(task_id, min(progress, 85), "assigning teams")

        # Assigning ball to player function
        print("Assigning ball to players...")
        update_progress(task_id, 85, "assigning ball to players")
        player_assigner = PlayerBallAssigner()
        team_ball_control = []
        for frame_num, player_track in enumerate(tracks['players']):
            ball_bbox = tracks['ball'][frame_num][1]['bbox']
            assigned_player = player_assigner.assign_ball_to_player(
                player_track, ball_bbox)

            if assigned_player != -1:
                tracks['players'][frame_num][assigned_player]['has_ball'] = True
                # we getting tracks of players of frame numbers and then assign teamp player with team
                team_ball_control.append(
                    tracks['players'][frame_num][assigned_player]['team'])
            else:
                # last person who has the ball
                if len(team_ball_control) > 0:
                    team_ball_control.append(team_ball_control[-1])
                else:
                    # default to team 1 if no previous control
                    team_ball_control.append(1)
        team_ball_control = np.array(team_ball_control)

        # Calling draw output function for object tracks
        print("Drawing annotations...")
        update_progress(task_id, 90, "drawing annotations")
        output_video_frames = tracker.draw_annotations(
            video_frames, tracks, team_ball_control)

        # Drawing camera movement
        print("Drawing camera movement...")
        update_progress(task_id, 92, "drawing camera movement")
        output_video_frames = camera_movement_estimator.draw_camera_movement(
            output_video_frames, camera_movement_per_frame)

        # Draw Speed and Distance
        print("Drawing speed and distance...")
        update_progress(task_id, 95, "drawing speed and distance")
        speed_and_distance_estimator.draw_speed_and_distance(
            output_video_frames, tracks)

        # Saving video
        print(f"Saving video to {output_video_path}...")
        update_progress(task_id, 98, "saving video")
        save_video(output_video_frames, output_video_path)
        print("Processing complete!")
        update_progress(task_id, 100, "completed")
        return True
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        update_progress(task_id, 0, "failed")
        return False

def main():
    # Define paths for Windows environment
    # Using one of the available videos from uploads directory
    input_video_path = 'uploads/48526c7f-298c-45cd-9a50-99e910153290.mp4'
    model_path = 'models/best.pt'
    track_stubs_path = 'stubs/track_stubs.pkl'
    camera_movement_stub_path = 'stubs/camera_movement_stub.pkl'
    output_video_path = 'output_videos/output_video.avi'
    
    # Create output directory if it doesn't exist
    os.makedirs('output_videos', exist_ok=True)
    
    # Check if input video exists
    if not os.path.exists(input_video_path):
        print(f"Error: Input video file not found at {input_video_path}")
        return

    # Reading video
    print("Reading video...")
    video_frames = read_video(input_video_path)
    
    # Check if video was read successfully
    if not video_frames:
        print(f"Error: Failed to read video frames from {input_video_path}")
        print("Please check if the video file is valid and accessible.")
        return
    
    print(f"Successfully read {len(video_frames)} frames from video")

    # Initializing Tracker
    print("Initializing tracker...")
    tracker = Tracker(model_path)

    # Getting object tracks
    print("Getting object tracks...")
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path=track_stubs_path)

    # Getting object positions
    print("Adding position to tracks...")
    tracker.add_position_to_tracks(tracks)

    # Adding camera movement estimator
    print("Estimating camera movement...")
    # Initializing by first frame
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                              read_from_stub=True,
                                                                              stub_path=camera_movement_stub_path)
    # Calling adjust camera position
    camera_movement_estimator.add_adjust_positions_to_tracks(
        tracks, camera_movement_per_frame)

    # Added View Transformer
    print("Transforming view...")
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolating/inserting ball positions
    print("Interpolating ball positions...")
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Adding speed and distance estimator
    print("Estimating speed and distance...")
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assigning player teams
    print("Assigning team colors...")
    team_assigner = TeamAssigner()  # initialising team assigner
    # assigning team their colours and by giving them first frame and tracks of player in first frame
    team_assigner.assign_team_color(video_frames[0],
                                    tracks['players'][0])

    # Looping over each player in each frame and assigning them to colour team
    print("Assigning players to teams...")
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Assigning ball to player function
    print("Assigning ball to players...")
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(
            player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            # we getting tracks of players of frame numbers and then assign teamp player with team
            team_ball_control.append(
                tracks['players'][frame_num][assigned_player]['team'])
        else:
            # last person who has the ball
            if len(team_ball_control) > 0:
                team_ball_control.append(team_ball_control[-1])
            else:
                # default to team 1 if no previous control
                team_ball_control.append(1)
    team_ball_control = np.array(team_ball_control)

    # Calling draw output function for object tracks
    print("Drawing annotations...")
    output_video_frames = tracker.draw_annotations(
        video_frames, tracks, team_ball_control)

    # Drawing camera movement
    print("Drawing camera movement...")
    output_video_frames = camera_movement_estimator.draw_camera_movement(
        output_video_frames, camera_movement_per_frame)

    # Draw Speed and Distance
    print("Drawing speed and distance...")
    speed_and_distance_estimator.draw_speed_and_distance(
        output_video_frames, tracks)

    # Saving video
    print(f"Saving video to {output_video_path}...")
    save_video(output_video_frames, output_video_path)
    print("Processing complete!")

if __name__ == "__main__":
    main()