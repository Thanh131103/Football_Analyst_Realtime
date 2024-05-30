import numpy as np  
import cv2
class ViewTransnformer():
    def __init__(self) :
        court_width=68
        court_length=23.32

        self.pixel_verticies = np.array([
            [110,1035],
            [265,275],
            [910,260],
            [1640,915]
        ])

        self.target_verticies = np.array([
            [0,court_width],
            [0,0],
            [court_length,0],
            [court_length,court_width]
        ])
        self.pixel_verticies = self.pixel_verticies.astype(np.float32)
        self.target_verticies = self.target_verticies.astype(np.float32)

        self.perspective_transform = cv2.getPerspectiveTransform(self.pixel_verticies, self.target_verticies)
    def transform_point(self,point):
        p=(int(point[0]),int(point[1]))
        is_inside=cv2.pointPolygonTest(self.pixel_verticies,p,False)>=0
        if not is_inside:
            return None
        reshape_point=point.reshape(-1,1,2).astype(np.float32)
        transform_point=cv2.perspectiveTransform(reshape_point,self.perspective_transform)
        return transform_point.reshape(-1,2)
    
    def add_transformed_position_to_tracks(self,tracks):
        for object,object_track in tracks.items():
            for frame_num,track in enumerate(object_track):
                for track_id,track_info in track.items():
                    positions = track_info['position_adjusted']
                    positions = np.array(positions)
                    positions_transformed=self.transform_point(positions)
                    if positions_transformed is not None:
                        positions_transformed=positions_transformed.squeeze().tolist()
                    tracks[object][frame_num][track_id]['positions_transformed']=positions_transformed
        return tracks