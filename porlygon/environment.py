from gym.spaces import text
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

from constants import VERTICES_PER_POLYGON, WINDOW_SIZE
from errors import MissingDependency

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
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps" : 60}

    def __init__(self, image_shape, image_path, max_step, render_mode=None):
        super(DrawPolygonEnv, self).__init__()

        self.img_shape = image_shape
        self.img_path = image_path
        self.max_step = max_step

        if render_mode == 'none':
            render_mode = None
        if render_mode != None:
            if PYGAME_IMPORT_ERROR is not None:
                # Want to render environment without installing pygame
                raise MissingDependency(
                    f"{PYGAME_IMPORT_ERROR}. Package pygame is required to render the environment, run `pip install porlygon[render]` to install it"
                )
            if render_mode not in self.metadata['render.modes']:
                # Want to render environment with unsupported render modes
                raise NotImplementedError(
                    f"Only {self.metadata['render.modes']} are supported render modes, but {render_mode} is provided")

        self.render_mode = render_mode
        # Define action and observation space
        # 2 numbers (x, y) for each vertix and 4 numbers for RGBA
        input_size = VERTICES_PER_POLYGON * 2 + 4
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(
            input_size, ), dtype=np.float32)
        # Two images for the reference image and the canvas
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=self.img_shape + (2,), dtype=np.uint8)
        self.dataset = self._load_dataset()

    def reset(self, seed = None):
        super().reset(seed=seed)
        # randomly select an image as reference
        self._ref_img = self.dataset[self.np_random.integers(
            len(self.dataset))]
        # reset the canvas as a white empty one
        self._canvas = torch.full(self.img_shape, 255)
        # restart from 0th step
        self._step_cnt = 0
        return self._get_obs()

    def step(self, action: np.ndarray):
        # apply action
        vertices = action[:-4].reshape(2, -1)
        rgb = np.round(action[-4:-1] * 255).astype(int)
        alpha = action[-1]
        rr, cc = skdraw.polygon(vertices[1:], vertices[2:])
        # alpha blending: new_color = (alpha)*(foreground_color) + (1 - alpha)*(background_color)
        # see https://graphics.fandom.com/wiki/Alpha_blending
        self._canvas[:, rr, cc] *= 1 - alpha
        self._canvas[:, rr, cc] += rgb * alpha
        # Calculate reward
        reward = structural_similarity_index_measure(
            self._canvas, self._ref_img, data_range=255)
        # Is max step reached?
        self._step_cnt += 1
        done = (self._step_cnt == self.max_step)
        # Optionally we can pass additional info, we are not using that for now
        return self._get_obs(), reward, done, self._get_info()

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
            self.font = freetype.SysFont(freetype.get_default_font(), 18)
      
        step_cnt_text = f"step: {self._step_cnt}"
        info_surf, info_rect= self.font.render(step_cnt_text, (220,0,0))
        self.screen.blit(info_surf, (0,0))
        text_height = info_rect.height * 1.2

        # Images should both fit in the window.
        image_top = text_height
        image_mid = WINDOW_SIZE[0] / 2
        image_size = min(image_mid, WINDOW_SIZE[1] - text_height)
        if self._step_cnt == 0:
            ref_surf = pygame.surfarray.make_surface(self._ref_img.numpy())
            ref_surf = pygame.transform.scale(ref_surf, (image_size, image_size))
            self.screen.blit(ref_surf, (0, image_top))

        canvas_surf = pygame.surfarray.make_surface(self._canvas.numpy())
        canvas_surf = pygame.transform.scale(canvas_surf, (image_size, image_size))
        self.screen.blit(canvas_surf, (image_mid, image_top))

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
        return torch.stack([self._ref_img, self._canvas], dim=-1)

    def _get_info(self):
        return {"step": self._step_cnt}

    def _load_dataset(self):
        transforms = tsfm.Compose(
            [tsfm.Resize(self.img_shape), tsfm.ToTensor()])
        return EnvironmentDataset(img_dir=self.img_path, transforms=transforms)


class EnvironmentDataset(Dataset):
    def __init__(self, img_dir, transforms=None):
        self.dir = img_dir
        self.transforms = transforms
        self.img_files = glob.glob(os.path.join(self.dir, '*.jpg'))

    def __getitem__(self, idx) -> torch.Tensor:
        image = read_image(self.img_files[idx])
        if self.transforms:
            image = self.transforms(image)
        return image

    def __len__(self):
        return len(self.img_files)
