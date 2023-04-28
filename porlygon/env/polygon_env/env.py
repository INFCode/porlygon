from typing import TYPE_CHECKING, Dict, Tuple, Optional, Any

import numpy as np
import skimage.draw as skdraw
import gym
from gym import spaces
import torch
import torchvision.transforms as tsfm
from torchmetrics.functional import structural_similarity_index_measure

from porlygon.env.constants import VERTICES_PER_POLYGON, WINDOW_SIZE, IMG_SHAPE
from porlygon.errors import MissingDependency
from porlygon.env.polygon_env import dataset

try:
    import pygame
    from pygame import freetype
except ImportError as e:
    PYGAME_IMPORT_ERROR = e
else:
    PYGAME_IMPORT_ERROR = None

if TYPE_CHECKING:
    import pygame


class PygameRenderManager:
    """
    A class to manage rendering images using pygame.

    Attributes:
        render_mode (str): The render mode to use, either "human" or "rgb_array".
        fps (int): The frame rate of the rendering in frames per second.
    """

    render_mode: str
    fps: int
    _screen: pygame.surface.Surface | pygame.Surface
    _clock: pygame.time.Clock
    _font: pygame.freetype.Font
    _text_area_height: float
    _info_bg_surf: pygame.Surface

    def __init__(self, render_mode: str, fps: int) -> None:
        """
        Initialize the PygameRenderManager with the given render mode and frame rate.

        Args:
            render_mode (str): The render mode to use, currently support "human" or "rgb_array".
            fps (int): The frame rate of the rendering in frames per second.
        """
        # User want to use rgb_array or human render mode
        pygame.init()
        self.render_mode = render_mode
        if self.render_mode == "human":
            pygame.display.init()
            self._screen = pygame.display.set_mode(WINDOW_SIZE)
        else:  # mode == "rgb_array"
            self._screen = pygame.Surface(WINDOW_SIZE)
        self._clock = pygame.time.Clock()
        font_size = 30
        self._font = freetype.SysFont("monospace", font_size)
        self._text_area_height = self._font.get_sized_height(font_size) * 1.5
        self._info_bg_surf = pygame.Surface((WINDOW_SIZE[0], self._text_area_height))
        self.fps = fps

    def _prepare_image(
        self, image: np.ndarray, size: Tuple[int, int]
    ) -> pygame.surface.Surface:
        image_surf = pygame.surfarray.make_surface(np.transpose(image, (2, 1, 0)))
        image_surf = pygame.transform.scale(image_surf, size)
        return image_surf

    def render(
        self, step_cnt: int, ref_img: np.ndarray, canvas: np.ndarray
    ) -> None | np.ndarray:
        """
        Render the given images and text to the screen.

        Args:
            step_cnt (int): The current step count.
            ref_img (np.ndarray): The reference image to be displayed.
            canvas (np.ndarray): The canvas image to be displayed.

        Returns:
            np.ndarray: The rendered image as an RGB array if render_mode is "rgb_array", otherwise None.
        """
        step_cnt_text = f"step: {step_cnt}"
        info_surf, _ = self._font.render(step_cnt_text, (220, 0, 0), pygame.SRCALPHA)

        # Images should both fit in the window.
        image_top = self._text_area_height
        image_mid = WINDOW_SIZE[0] / 2
        image_side_length = int(min(image_mid, WINDOW_SIZE[1] - self._text_area_height))
        image_size = image_side_length, image_side_length

        # canvas_surf = pygame.surfarray.make_surface(np.transpose(canvas, (2, 1, 0)))
        # canvas_surf = pygame.transform.scale(canvas_surf, (image_size, image_size))
        canvas_surf = self._prepare_image(canvas, image_size)
        if step_cnt == 0:
            # ref_surf = pygame.surfarray.make_surface(np.transpose(ref_img, (2, 1, 0)))
            # ref_surf = pygame.transform.scale(ref_surf, (image_size, image_size))
            ref_surf = self._prepare_image(ref_img, image_size)
            self._screen.blit(ref_surf, (0, image_top))
        self._screen.blit(canvas_surf, (image_mid, image_top))
        self._info_bg_surf.fill((0, 0, 0))
        self._info_bg_surf.blit(info_surf, (0, 0))
        self._screen.blit(self._info_bg_surf, (0, 0))

        if self.render_mode == "human":
            pygame.event.pump()
            self._clock.tick(self.fps)
            pygame.display.flip()
            return None

        if self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self._screen)), axes=(1, 0, 2)
            )

        return None

    def close(self) -> None:
        """
        Close the pygame window and clean up resources.
        """
        if self.render_mode == "human":
            pygame.display.quit()
        pygame.quit()


class DrawPolygonEnv(gym.Env):
    """
    A custom OpenAI Gym environment to draw polygons on a canvas to match a reference image.

    Args:
        image_shape (tuple): The shape of the input images (default: IMG_SHAPE).
        image_path (str): The path to the directory containing the input images
            (default: "data/jpg/").
        max_step (int): The maximum number of steps allowed in each episode (default: 100).
        render_mode (str): The rendering mode. Can be "human" or "rgb_array" (default: None).

    Attributes:
        action_space (gym.spaces.Box): The action space for the environment.
        observation_space (gym.spaces.Dict): The observation space for the environment.
        metadata (dict): Additional metadata for the environment.
    """

    metadata: Dict[str, Any] = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }

    action_space: spaces.Box
    observation_space: spaces.Dict

    img_shape: Tuple[int, int, int]
    img_path: str
    max_step: int

    render_manager: Optional[PygameRenderManager]

    data_set: dataset.EnvironmentDataset

    def __init__(
        self,
        image_shape=IMG_SHAPE,
        image_path="data/jpg/",
        max_step=100,
        render_mode=None,
    ):
        super(DrawPolygonEnv, self).__init__()

        self.img_shape = image_shape
        self.img_path = image_path
        self.max_step = max_step

        # render setup
        if render_mode == "none":
            render_mode = None
        if render_mode is not None:
            if render_mode not in self.metadata["render_modes"]:
                # Want to render environment with unsupported render modes
                raise NotImplementedError(
                    f"Only {self.metadata['render_modes']} are supported render modes, "
                    "but {render_mode} is provided"
                )
            if PYGAME_IMPORT_ERROR is not None:
                # Want to render environment without installing pygame
                raise MissingDependency(
                    f"{PYGAME_IMPORT_ERROR}. Package pygame is required to render "
                    "the environment, run `pip install porlygon[render]` to install it"
                )
            self.render_manager = PygameRenderManager(
                render_mode, self.metadata["render_fps"]
            )

        # Define action and observation space
        # 2 numbers (x, y) for each vertix and 4 numbers for RGBA
        input_size = VERTICES_PER_POLYGON * 2 + 4
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(input_size,), dtype=np.float32
        )
        # Two images for the reference image and the canvas
        self.observation_space = spaces.Dict(
            {
                "reference": spaces.Box(
                    low=0, high=255, shape=self.img_shape, dtype=np.uint8
                ),
                "canvas": spaces.Box(
                    low=0, high=255, shape=self.img_shape, dtype=np.uint8
                ),
            }
        )
        self.data_set = self._load_dataset()

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state.

        Args:
            seed (int): The random seed used to generate the random state (default: None).
            options (dict): Additional options (not used in this function).

        Returns:
            tuple: A tuple containing the initial observation and information.
        """
        # let gym.Env to handle the seeding
        super().reset(seed=seed)
        # options is not used here
        del options
        # randomly select an image as reference
        self._ref_img = self.data_set[
            self.np_random.integers(len(self.data_set))
        ].numpy()
        # reset the canvas as a white empty one
        self._canvas = np.full(self.img_shape, 255, dtype=np.uint8)
        # restart from 0th step
        self._step_cnt = 0
        return (self._get_obs(), self._get_info())

    def step(self, action: np.ndarray):
        """
        Executes one step in the environment.

        Args:
            action (np.ndarray): The action to take. The indicies are interpret in the
            order of: all x positions (V in total) -> all y positions (V in total) ->
            rgba (4 in total), where V is the number of vertices

        Returns:
            tuple: A tuple containing the new observation, the reward, the termination status,
            the truncation status, and additional information.
        """
        # apply action
        vertices = action[:-4].reshape(2, -1)
        rgb = np.round(action[-4:-1] * 255).astype(int)
        alpha = action[-1]
        rr, cc = skdraw.polygon(
            vertices[0] * (self.img_shape[1] - 1),
            vertices[1] * (self.img_shape[2] - 1),
        )

        # alpha blending: new_color = (alpha)*(foreground_color) + (1 - alpha)*(background_color)
        # see https://graphics.fandom.com/wiki/Alpha_blending
        self._canvas[:, rr, cc] = (
            self._canvas[:, rr, cc] * (1 - alpha) + np.expand_dims(rgb, 1) * alpha
        ).astype(int)
        # Calculate reward
        reward = structural_similarity_index_measure(
            torch.from_numpy(self._canvas).unsqueeze(0).to(torch.float),
            torch.from_numpy(self._ref_img).unsqueeze(0).to(torch.float),
            data_range=255,
        )
        # There should not be a second return value
        assert isinstance(reward, torch.Tensor)
        reward = reward.item()
        # Is max step reached? Terminate if so.
        self._step_cnt += 1
        term = self._step_cnt == self.max_step
        # This environment should never be truncated
        trunc = False
        return self._get_obs(), reward, term, trunc, self._get_info()

    def render(self):
        """
        Renders the environment.

        Returns:
            numpy.ndarray or None: The rendered image as a numpy array (if render_mode
            is "rgb_array"), or None.
        """
        if self.render_manager is None:
            return

        self.render_manager.render(self._step_cnt, self._ref_img, self._canvas)

    def close(self):
        """
        Closes the environment.
        """
        if self.render_manager is not None:
            self.render_manager.close()

    def _get_obs(self):
        """
        Returns the current observation.

        Returns:
            dict: A dictionary containing the reference image and the canvas.
        """
        return {"reference": self._ref_img, "canvas": self._canvas}

    def _get_info(self):
        """
        Returns additional information about the environment.

        Returns:
            dict: A dictionary containing the current step count.
        """
        return {"step": self._step_cnt}

    def _load_dataset(self):
        """
        Loads the input images into memory.

        Returns:
            Dataset: A PyTorch Dataset containing the input images.
        """
        transforms = tsfm.Compose([tsfm.Resize(self.img_shape[1:])])
        return dataset.EnvironmentDataset(img_dir=self.img_path, transforms=transforms)
