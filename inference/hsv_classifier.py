import copy
from typing import List, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np

from inference.base_classifier import BaseClassifier
from inference.colors import all_colors


class HSVClassifier(BaseClassifier):
    def __init__(self, filters: List[dict]):
        """
        Initialize HSV Classifier

        Parameters
        ----------
        filters: List[dict]
            List of colors to classify

            Format:
            [
                {
                    "name": "Boca Juniors",
                    "colors": [inferece.colors.blue, inference.colors.yellow],
                },
                {
                    "name": "River Plate",
                    "colors": [inference.colors.red, inference.colors.white],
                },
                {
                    "name": "Real Madrid",
                    "colors": [inference.colors.white],
                },
                {
                    "name": "Barcelona",
                    "colors": [custom_color],
                },
            ]

            If you want to add a specific color, you can add it as a Python dictionary with the following format:

            custom_color = {
                "name":"my_custom_color",
                "lower_hsv": (0, 0, 0),
                "upper_hsv": (179, 255, 255)
            }

            You can find your custom hsv range with an online tool like https://github.com/hariangr/HsvRangeTool
        """
        super().__init__()

        self.filters = [self.check_filter_format(color_filter) for color_filter in filters]

    def check_tuple_format(self, a_tuple: tuple, name: str) -> None:
        """
        Check tuple format

        Parameters
        ----------
        a_tuple : tuple
            Tuple to check
        name : str
            Name of the tuple

        Returns
        -------
        tuple
            Tuple checked

        Raises
        ------
        ValueError
            If tuple is not a tuple
        ValueError
            If tuple is not a tuple of 3 elements
        ValueError
            If tuple elements are not integers
        """
        # Check class is a tuple
        if type(a_tuple) != tuple:
            raise ValueError(f"{name} must be a tuple")

        # Check length 3
        if len(a_tuple) != 3:
            raise ValueError(f"{name} must be a tuple of length 3")

        # Check all values are ints
        for value in a_tuple:
            if type(value) != int:
                raise ValueError(f"{name} values must be ints")

    def check_tuple_intervals(self, a_tuple: tuple, name: str):
        """
        Check tuple intervals

        Parameters
        ----------
        a_tuple : tuple
            Tuple to check
        name : str
            Name of the tuple

        Raises
        ------
        ValueError
            If first element is not between 0 and 179
        ValueError
            If second element is not between 0 and 255
        ValueError
            If third element is not between 0 and 255
        """

        # check hue is between 0 and 179
        if a_tuple[0] < 0 or a_tuple[0] > 179:
            raise ValueError(f"{name} hue must be between 0 and 179")

        # check saturation is between 0 and 255
        if a_tuple[1] < 0 or a_tuple[1] > 255:
            raise ValueError(f"{name} saturation must be between 0 and 255")

        # check value is between 0 and 255
        if a_tuple[2] < 0 or a_tuple[2] > 255:
            raise ValueError(f"{name} value must be between 0 and 255")

    def check_color_format(self, color: dict) -> dict:
        """
        Check color format

        Parameters
        ----------
        color : dict
            Color to check

        Returns
        -------
        dict
            Color checked

        Raises
        ------
        ValueError
            If color is not a dict
        ValueError
            If color does not have a name
        ValueError
            If color name is not a string
        ValueError
            If color does not have a lower hsv
        ValueError
            If color does not have an upper hsv
        ValueError
            If lower hsv doesnt have correct tuple format
        ValueError
            If upper hsv doesnt have correct tuple format
        """

        if type(color) != dict:
            raise ValueError("Color must be a dict")
        if "name" not in color:
            raise ValueError("Color must have a name")
        if type(color["name"]) != str:
            raise ValueError("Color name must be a string")
        if "lower_hsv" not in color:
            raise ValueError("Color must have a lower hsv")
        if "upper_hsv" not in color:
            raise ValueError("Color must have an upper hsv")

        self.check_tuple_format(color["lower_hsv"], "lower_hsv")
        self.check_tuple_format(color["upper_hsv"], "upper_hsv")

        self.check_tuple_intervals(color["lower_hsv"], "lower_hsv")
        self.check_tuple_intervals(color["upper_hsv"], "upper_hsv")

        return color

    def check_filter_format(self, color_filter: dict) -> dict:
        """
        Check filter format

        Parameters
        ----------
        color_filter : dict
            Filter to check

        Returns
        -------
        dict
            Filter checked

        Raises
        ------
        ValueError
            If filter is not a dict
        ValueError
            If filter does not have a name
        ValueError
            If filter does not have colors
        ValueError
            If filter colors is not a list or a tuple
        """

        if type(color_filter) != dict:
            raise ValueError("Filter must be a dict")
        if "name" not in color_filter:
            raise ValueError("Filter must have a name")
        if "colors" not in color_filter:
            raise ValueError("Filter must have colors")

        if type(color_filter["name"]) != str:
            raise ValueError("Filter name must be a string")

        if type(color_filter["colors"]) != list and type(color_filter["colors"] != tuple):
            raise ValueError("Filter colors must be a list or tuple")

        color_filter["colors"] = [
            self.check_color_format(color) for color in color_filter["colors"]
        ]

        return color_filter

    def get_hsv_img(self, img: np.ndarray) -> np.ndarray:
        """
        Get HSV image

        Parameters
        ----------
        img : np.ndarray
            Image to convert

        Returns
        -------
        np.ndarray
            HSV image
        """
        return cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)

    def apply_filter(self, img: np.ndarray, color_filter: dict) -> np.ndarray:
        """
        Apply filter to image

        Parameters
        ----------
        img : np.ndarray
            Image to apply filter to
        color_filter : dict
            Filter to apply

        Returns
        -------
        np.ndarray
            Filtered image
        """
        img_hsv = self.get_hsv_img(img)
        mask = cv2.inRange(img_hsv, color_filter["lower_hsv"], color_filter["upper_hsv"])
        return cv2.bitwise_and(img, img, mask=mask)

    def crop_img_for_jersey(self, img: np.ndarray) -> np.ndarray:
        """
        Crop image to get only the jersey part

        Parameters
        ----------
        img : np.ndarray
            Image to crop

        Returns
        -------
        np.ndarray
            Cropped image
        """
        height, width, _ = img.shape

        y_start = int(height * 0.15)
        y_end = int(height * 0.50)
        x_start = int(width * 0.15)
        x_end = int(width * 0.85)

        return img[y_start:y_end, x_start:x_end]

    def add_median_blur(self, img: np.ndarray) -> np.ndarray:
        """
        Add median blur to image

        Parameters
        ----------
        img : np.ndarray
            Image to add blur to

        Returns
        -------
        np.ndarray
            Blurred image
        """
        return cv2.medianBlur(img, 5)

    def non_black_pixels_count(self, img: np.ndarray) -> float:
        """
        Returns the amount of non black pixels an image has

        Parameters
        ----------
        img : np.ndarray
            Image

        Returns
        -------
        float
            Count of non black pixels in img
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.countNonZero(img)

    def crop_filter_and_blur_img(self, img: np.ndarray, color_filter: dict) -> np.ndarray:
        """
        Crops image to get only the jersey part. Filters the colors and adds a median blur.

        Parameters
        ----------
        img : np.ndarray
            Image to crop
        color_filter : dict
            Filter to apply

        Returns
        -------
        np.ndarray
            Cropped image
        """
        transformed_img = img.copy()
        transformed_img = self.crop_img_for_jersey(transformed_img)
        transformed_img = self.apply_filter(transformed_img, color_filter)
        transformed_img = self.add_median_blur(transformed_img)
        return transformed_img

    def add_non_black_pixels_count_in_filter(
        self, img: np.ndarray, color_filter: dict
    ) -> dict:
        """
        Applies filter to image and saves the number of non black pixels in the filter.

        Parameters
        ----------
        img : np.ndarray
            Image to apply filter to
        color_filter : dict
            Filter to apply to img

        Returns
        -------
        dict
            Filter with non black pixels count
        """
        transformed_img = self.crop_filter_and_blur_img(img, color_filter)
        color_filter["non_black_pixels_count"] = self.non_black_pixels_count(transformed_img)
        return color_filter

    def predict_img(self, img: np.ndarray) -> str:
        """
        Gets the filter with most non blakc pixels on img and returns its name.

        Parameters
        ----------
        img : np.ndarray
            Image to predict

        Returns
        -------
        str
            Name of the filter with most non black pixels on img
        """
        if img is None:
            raise ValueError("Image can't be None")

        filters = copy.deepcopy(self.filters)

        for i, color_filter in enumerate(filters):
            for color in color_filter["colors"]:
                color = self.add_non_black_pixels_count_in_filter(img, color)
                if "non_black_pixels_count" not in color_filter:
                    color_filter["non_black_pixels_count"] = 0
                color_filter["non_black_pixels_count"] += color["non_black_pixels_count"]

        max_non_black_pixels_filter = max(
            filters, key=lambda x: x["non_black_pixels_count"]
        )

        return max_non_black_pixels_filter["name"]

    def predict(self, input_image: Union[np.ndarray, List[np.ndarray]]) -> str:
        """
        Predicts the name of the team from the input image or a list of input images.

        Parameters
        ----------
        input_image : Union[np.ndarray, List[np.ndarray]]
            Image or list of images to predict

        Returns
        -------
        str
            Predicted team name
        """

        if isinstance(input_image, list):
            predictions = [self.predict_img(img) for img in input_image]
            return ', '.join(predictions)
        else:
            return self.predict_img(input_image)

    def transform_image_for_every_color(self, img: np.ndarray, colors: List[dict] = None) -> List[dict]:
        """
        Transforms image for every color in every filter.

        Parameters
        ----------
        img : np.ndarray
            Image to transform
        colors : List[dict], optional
            List of colors to transform image for, by default None

        Returns
        -------
        List[dict]
            List of Transformed images

            [
                {
                    "name": "red",
                    "image": transformed_image,
                    "original": img,
                },
                {
                    "name": "blue",
                    "image": transformed_image,
                    "original": img,
                },
                # ... other colors ...
            ]
        """
        transformed_images = []

        colors_to_transform = all_colors
        if colors:
            colors_to_transform = colors

        for color in colors_to_transform:
            transformed_image = self.crop_filter_and_blur_img(img, color)
            transformed_images.append({
                "name": color["name"],
                "image": transformed_image,
                "original": img,
            })

        return transformed_images

    def plot_every_color_output(self, img: np.ndarray, colors: List[dict] = None, save_img_path: str = None) -> None:
        """
            Plots every color output of the image.

            Parameters
            ----------
            img : np.ndarray
                Image to plot
            colors : List[dict], optional
                List of colors to plot, by default None
            save_img_path : str, optional
                Path to save image to, by default None
            """
        transformed_images = self.transform_image_for_every_color(img, colors)
        transformed_images[0]["original"] = img

        n = len(transformed_images)

        fig, axs = plt.subplots(1, n, figsize=(n * 5, 5))

        fig.suptitle("Every color output")
        for i, color_info in enumerate(transformed_images):
            key = color_info["name"]
            value = cv2.cvtColor(color_info["image"], cv2.COLOR_BGR2RGB)
            axs[i].imshow(value)
            if key == "original":
                axs[i].set_title(f"{key}")
            else:
                gray_img = cv2.cvtColor(color_info["image"], cv2.COLOR_BGR2GRAY)
                power = cv2.countNonZero(gray_img)
                axs[i].set_title(f"{key}: {power}")
        plt.show()

        if save_img_path is not None:
            fig.savefig(save_img_path)
