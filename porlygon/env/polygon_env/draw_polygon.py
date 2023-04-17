import numpy as np
import skimage.draw as skdraw
import glob
import os
import gym
from gym import spaces
import torch
import torchvision.transforms as tsfm
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchmetrics.functional import structural_similarity_index_measure

from porlygon.env.constants import VERTICES_PER_POLYGON, WINDOW_SIZE, IMG_SHAPE
from porlygon.errors import MissingDependency

try:
    import pygame
    from pygame import freetype
except ImportError as e:
    PYGAME_IMPORT_ERROR = e
else:
    PYGAME_IMPORT_ERROR = None


class DrawPolygonEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env where the agent must learn to go always left.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

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
        self.screen = None
        self.clock = None
        self.font = None

        if render_mode == "none":
            render_mode = None
        if render_mode != None:
            if PYGAME_IMPORT_ERROR is not None:
                # Want to render environment without installing pygame
                raise MissingDependency(
                    f"{PYGAME_IMPORT_ERROR}. Package pygame is required to render the environment, run `pip install porlygon[render]` to install it"
                )
            if render_mode not in self.metadata["render_modes"]:
                # Want to render environment with unsupported render modes
                raise NotImplementedError(
                    f"Only {self.metadata['render_modes']} are supported render modes, but {render_mode} is provided"
                )

        self.render_mode = render_mode
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
        self.dataset = self._load_dataset()

    def reset(self, seed=None, options=None):
        # let gym.Env to handle the seeding
        super().reset(seed=seed)
        # options is not used here
        del options
        # randomly select an image as reference
        self._ref_img = self.dataset[self.np_random.integers(len(self.dataset))].numpy()
        # reset the canvas as a white empty one
        self._canvas = np.full(self.img_shape, 255, dtype=np.uint8)
        # restart from 0th step
        self._step_cnt = 0
        return (self._get_obs(), self._get_info())

    def step(self, action: np.ndarray):
        # apply action
        vertices = action[:-4].reshape(2, -1)
        rgb = np.round(action[-4:-1] * 255).astype(int)
        alpha = action[-1]
        rr, cc = skdraw.polygon(
            vertices[
                0,
            ]
            * self.img_shape[1],
            vertices[
                1,
            ]
            * self.img_shape[2],
        )
        # alpha blending: new_color = (alpha)*(foreground_color) + (1 - alpha)*(background_color)
        # see https://graphics.fandom.com/wiki/Alpha_blending

        self._canvas[:, rr, cc] = (
            self._canvas[:, rr, cc] * (1 - alpha) + np.expand_dims(rgb * alpha, 1)
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
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(WINDOW_SIZE)
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface(WINDOW_SIZE)
        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.font is None:
            font_size = 30
            self.font = freetype.SysFont("monospace", font_size)
            self.text_area_height = self.font.get_sized_height(font_size) * 1.5
            self.info_bg_surf = pygame.Surface((WINDOW_SIZE[0], self.text_area_height))

        step_cnt_text = f"step: {self._step_cnt}"
        info_surf, _ = self.font.render(step_cnt_text, (220, 0, 0), pygame.SRCALPHA)

        # Images should both fit in the window.
        image_top = self.text_area_height
        image_mid = WINDOW_SIZE[0] / 2
        image_size = min(image_mid, WINDOW_SIZE[1] - self.text_area_height)

        canvas_surf = pygame.surfarray.make_surface(
            np.transpose(self._canvas, (2, 1, 0))
        )
        canvas_surf = pygame.transform.scale(canvas_surf, (image_size, image_size))
        if self._step_cnt == 0:
            ref_surf = pygame.surfarray.make_surface(
                np.transpose(self._ref_img, (2, 1, 0))
            )
            ref_surf = pygame.transform.scale(ref_surf, (image_size, image_size))
            self.screen.blit(ref_surf, (0, image_top))
        self.screen.blit(canvas_surf, (image_mid, image_top))
        self.info_bg_surf.fill((0, 0, 0))
        self.info_bg_surf.blit(info_surf, (0, 0))
        self.screen.blit(self.info_bg_surf, (0, 0))

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

    def _get_obs(self):
        return {"reference": self._ref_img, "canvas": self._canvas}

    def _get_info(self):
        return {"step": self._step_cnt}

    def _load_dataset(self):
        transforms = tsfm.Compose([tsfm.Resize(self.img_shape[1:])])
        return EnvironmentDataset(img_dir=self.img_path, transforms=transforms)


class EnvironmentDataset(Dataset):
    def __init__(self, img_dir, transforms=None):
        self.dir = img_dir
        self.transforms = transforms
        self.img_files = glob.glob(os.path.join(self.dir, "*.jpg"))

    def __getitem__(self, idx):
        image = read_image(self.img_files[idx])
        if self.transforms:
            image = self.transforms(image)
        return image

    def __len__(self):
        return len(self.img_files)
