from utils import save_video,read_video
from trackers import Tracker
from team_assign import TeamAssigner
from player_ball_assign import PlayerBallAssigner
from camera_movement_track import CameraMovementEstimator
import numpy as np
from transformer import ViewTransnformer
from speed_distance_estimated import SpeedAndDistance_Estimator
def main():
    video_frames = read_video('data/08fd33_1.mp4')
    # print(video_frames)
    

    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_trackers(video_frames,read_from_stub=True,stub_path='stubs/track_stubs.pkl')

    tracker.add_position_to_tracks(tracks)

    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=True,
                                                                                stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)
    #Transform position to 2d 
    view_transform = ViewTransnformer()
    view_transform.add_transformed_position_to_tracks(tracks)
    #Speed and Distance
    speed_distance= SpeedAndDistance_Estimator()
    speed_distance.add_speed_and_distance_to_tracks(tracks)

    #BallPosition
    tracks["ball"]=tracker.interpolate_ball_position(tracks["ball"])
    #assign team
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])
    


    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    ##Assgign ball to player
    player_assigner = PlayerBallAssigner()
    team_ball_control=[]
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox=tracks["ball"][frame_num][1]["bbox"]
        assigner_player= player_assigner.assign_ball(player_track, ball_bbox)
        if assigner_player!=-1:
            tracks['players'][frame_num][assigner_player]['has_ball']=True
            team_ball_control.append(tracks["players"][frame_num][assigner_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control=np.array(team_ball_control)
        
    ##Draw tracks
    output_video_frames =tracker.draw_annotations(video_frames,tracks,team_ball_control)
    ##Draw camera movements
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)
    ##Draw Speed and Distance
    speed_distance.draw_speed_and_distance(output_video_frames,tracks)
    # Draw output 

    save_video(output_video_frames,'output_video/output_video1.avi')
if __name__ == '__main__':
    main()