'''
This is a simple example of color object tracking.
'''
import random as rng
import argparse
import cv2
import numpy as np
from enum import Enum

class ProcessingType(Enum):
    RAW = 0
    TRACKER = 1
    HUE = 2
    SATURATION = 3
    VALUE = 4
    MASK = 5


# MODEL
class ColorTracker:
    def __init__(self, video_path: str, hue_tolerance: int, saturation_tolerance: int, value_tolerance: int, tracked_color: None | tuple[int, int, int] = None) -> None:
        self._video = cv2.VideoCapture(video_path)
        if not self._video.isOpened():
            raise ValueError(f'Unable to open video at path {video_path}.')
        self._tracked_color = tracked_color
        self._hue_tolerance = hue_tolerance
        self._saturation_tolerance = saturation_tolerance
        self._value_tolerance = value_tolerance
        self._frame: None | np.ndarray = None
        self._processed_frame: None | np.ndarray = None
        self._processing_type: ProcessingType = ProcessingType.RAW

    def set_processing_type(self, ptype:ProcessingType)->None:
        self._processing_type = ptype

    def set_reference_color_by_position(self, x:int, y:int)-> None:
        hsv_frame:np.ndarray = cv2.cvtColor(self._frame, cv2.COLOR_RGB2HSV)
        self._tracked_color = hsv_frame[y,x,:]

    def update_frame(self)->bool:
        read_succesfull, self._frame = self._video.read()
        if read_succesfull:
            self._process_frame()
        return read_succesfull

    def _process_frame(self) -> None:
        if self._processing_type == ProcessingType.RAW:
            self._processed_frame = self._frame
            return
        hsv_frame: np.ndarray = cv2.cvtColor(self._frame, cv2.COLOR_RGB2HSV)
        hue = hsv_frame[:, :, 0]
        saturation = hsv_frame[:, :, 1]
        value = hsv_frame[:, :, 2]

        if self._processing_type in {ProcessingType.HUE, ProcessingType.SATURATION, ProcessingType.VALUE,
                                     ProcessingType.MASK} and self._tracked_color is None:
            raise ValueError('Attempted processing mode that requires a tracking color set without it set.')

        if self._processing_type == ProcessingType.HUE:
            self._processed_frame = hue
            return
        elif self._processing_type == ProcessingType.SATURATION:
            self._processed_frame = saturation
            return
        elif self._processing_type == ProcessingType.VALUE:
            self._processed_frame = value
            return

        lower_bound = np.array(
            [self._tracked_color[0] - self._hue_tolerance, self._tracked_color[1] - self._saturation_tolerance,
             self._tracked_color[2] - self._value_tolerance], dtype=np.int32)
        upper_bound = np.array(
            [self._tracked_color[0] + self._hue_tolerance, self._tracked_color[1] + self._saturation_tolerance,
             self._tracked_color[2] + self._value_tolerance], dtype=np.int32)
        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
        if self._processing_type == ProcessingType.MASK:
            self._processed_frame = mask

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            if self._processing_type == ProcessingType.TRACKER:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                drawing = np.zeros_like(self._frame, dtype=np.uint8)
                cv2.rectangle(drawing, (x, y), (x + w, y + h), (0, 255, 0), 2)
                self._processed_frame = cv2.addWeighted(self._frame, 0.8, drawing, 0.2, 0)
        else:
            self._processed_frame = self._frame.copy()

    def get_frame(self)->np.ndarray:
        if self._frame is None:
            raise ValueError(f'Attempted to get frame from uninitialized color tracker.')
        return self._frame.copy()

    def get_processed_frame(self)->np.ndarray:
        return self._processed_frame.copy()


# VIEW
class Display:

    def __init__(self, window_name:str)->None:
        self._window = cv2.namedWindow(window_name)
        self._window_name = window_name

    def update_display(self, image: np.ndarray)->None:
        cv2.imshow(self._window_name, image)

    def get_window_name(self)-> str:
        return self._window_name

# CONTROLER
class EventHandler:

    PROCESSING_TYPE_KEYMAP = {
        ord('h'): ProcessingType.HUE,
        ord('s'): ProcessingType.SATURATION,
        ord('v'): ProcessingType.VALUE,
        ord('r'): ProcessingType.RAW,
        ord('m'): ProcessingType.MASK,
        ord('t'): ProcessingType.TRACKER
    }

    def __init__(self, tracker: ColorTracker, display: Display, timeout: int)->None:
        self._window_name = display.get_window_name()
        self._tracker = tracker
        self._timeout = timeout
        cv2.setMouseCallback(self._window_name, self._handle_mouse)


    def _handle_mouse(self, event,x,y,flags,param)->None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self._tracker.set_reference_color_by_position(x,y)
            print(self._tracker._tracked_color)


    def _handle_keys(self)-> bool:
        keycode = cv2.waitKey(self._timeout)
        if keycode == ord('q') or keycode == 27:
            return False
        elif keycode in EventHandler.PROCESSING_TYPE_KEYMAP.keys():
            self._tracker.set_processing_type(EventHandler.PROCESSING_TYPE_KEYMAP[keycode])
        return True

    def handle_events(self)-> bool:
        return self._handle_keys()



def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-v', '--video_path', type=str, required=True, help='Path to video that will be processed.')
    parser.add_argument('-ht', '--hue_tolerance', type=int, required=True, help='Hue tolerance for color tracking.')
    parser.add_argument('-st', '--saturation_tolerance', type=int, required=True, help='Saturation tolerance for color tracking.')
    parser.add_argument('-vt', '--value_tolerance', type=int, required=True, help='Value tolerance for value (brightness) in color tracking.')
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    try:
        WINDOW_NAME = 'Color tracker'
        WAITKEY_TIMEOUT = 10
        tracker = ColorTracker(args.video_path, args.hue_tolerance, args.saturation_tolerance, args.value_tolerance)
        display = Display(WINDOW_NAME)
        event_handler = EventHandler(tracker, display, WAITKEY_TIMEOUT)
        while True:
            if not tracker.update_frame():
                break
            display.update_display(tracker.get_processed_frame())
            if not event_handler.handle_events():
                break
    except ValueError as e:
        print(e)


if __name__ == '__main__':
    main(parse_arguments())
