import argparse

import cv2
import numpy as np

from PIL import Image
from norfair import Tracker, Video
from norfair.camera_motion import MotionEstimator
from norfair.distances import mean_euclidean

from inference import Converter, HSVClassifier, InertiaClassifier, Yolo
from inference.filters import filters
from main_utils import (
    get_ball_detections,
    get_main_ball,
    get_player_detections,
    update_motion_estimator,
)
from soccer import Match, Player, Team
from soccer.draw import AbsolutePath
from soccer.pass_event import Pass


def run(video_path: str = "videos/example.mp4",
        ball_model_path: str = "models/ball.pt",
        passes: bool = False,
        possession: bool = False):
    video = Video(input_path=video_path)
    fps = video.video_capture.get(cv2.CAP_PROP_FPS)

    # Object Detectors
    player_detector = Yolo()
    ball_detector = Yolo(model_path=ball_model_path)

    # HSV Classifier
    hsv_classifier = HSVClassifier(filters=filters)

    # Add inertia to classifier
    classifier = InertiaClassifier(classifier=hsv_classifier, inertia=20)

    # TODO: Get the teams automatically (maybe from a csv or config file)
    # Teams and Match
    home_team = Team(
        name="Gimnasia (Jujuy)",
        abbreviation="GEJ",
        color=(192, 118, 255),
        board_color=(192, 118, 255),
        text_color=(0, 0, 0),
    )
    visitor_team = Team(
        name="Chacarita",
        abbreviation="CHA",
        color=(255, 255, 255)
    )
    teams = [home_team, visitor_team]
    match = Match(home=home_team, away=visitor_team, fps=fps)
    match.team_possession = home_team

    # Tracking
    player_tracker = Tracker(
        distance_function=mean_euclidean,
        distance_threshold=250,
        initialization_delay=3,
        hit_counter_max=90,
    )

    ball_tracker = Tracker(
        distance_function=mean_euclidean,
        distance_threshold=150,
        initialization_delay=20,
        hit_counter_max=2000,
    )
    motion_estimator = MotionEstimator()
    coord_transformations = None

    # Paths
    path = AbsolutePath()

    # Get Counter img
    possession_background = match.get_possession_background()
    passes_background = match.get_passes_background()

    for i, frame in enumerate(video):

        # Get Detections
        # players_detections = get_player_detections(player_detector, frame)
        ball_detections = get_ball_detections(ball_detector, frame)
        detections = ball_detections  # + players_detections TODO: uncomment this if you want to track players

        # Update trackers
        coord_transformations = update_motion_estimator(
            motion_estimator=motion_estimator,
            detections=detections,
            frame=frame,
        )

        # player_track_objects = player_tracker.update(
        #     detections=players_detections, coord_transformations=coord_transformations
        # )

        ball_track_objects = ball_tracker.update(
            detections=ball_detections, coord_transformations=coord_transformations
        )

        # player_detections = Converter.TrackedObjects_to_Detections(player_track_objects)
        ball_detections = Converter.TrackedObjects_to_Detections(ball_track_objects)

        # player_detections = classifier.predict_from_detections(
        #     detections=player_detections,
        #     img=frame,
        # )

        # Match update
        ball = get_main_ball(ball_detections)
        # players = Player.from_detections(detections=players_detections, teams=teams)
        # match.update(players, ball)
        match.update([], ball)

        # Draw
        pil_frame = Image.fromarray(frame)

        if possession:
            # pil_frame = Player.draw_players(
            #     players=players, frame=pil_frame, confidence=False, id=True
            # )

            pil_frame = path.draw(
                img=pil_frame,
                detection=ball.detection,
                coord_transformations=coord_transformations,
                color=match.team_possession.color,
            )

            # TODO: uncomment this if you want to draw the possession counter
            # pil_frame = match.draw_possession_counter(
            #     pil_frame, counter_background=possession_background, debug=False
            # )

            if ball:
                pil_frame = ball.draw(pil_frame)

        if passes:
            pass_list = match.passes

            pil_frame = Pass.draw_pass_list(
                img=pil_frame, passes=pass_list, coord_transformations=coord_transformations
            )

            # TODO: uncomment this if you want to draw the passes counter
            # pil_frame = match.draw_passes_counter(
            #     pil_frame, counter_background=passes_background, debug=False
            # )

        frame = np.array(pil_frame)

        # Write video
        video.write(frame)


if __name__ == "__main__":
    run(video_path="videos/prueba.mp4", ball_model_path="models/yolov5x.pt", passes=True, possession=True)
