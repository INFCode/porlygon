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

from constants import VERTICES_PER_POLYGON

class DrawPolygonEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env where the agent must learn to go always left. 
    """
    metadata = {'render.modes': []}

    def __init__(self, image_shape, image_path, max_step, render_mode='none'):
        super(DrawPolygonEnv, self).__init__()

        self.img_shape = image_shape
        self.img_path = image_path
        self.max_step = max_step 
        if render_mode not in self.metadata['render.modes']:
            raise NotImplementedError()
        self.render_mode = render_mode
        # Define action and observation space
        # They must be gym.spaces objects
        # 2 numbers (x, y) for each vertix and 4 numbers for RGBA
        input_size = VERTICES_PER_POLYGON * 2 + 4
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(
            input_size, ), dtype=np.float32)
        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=self.img_shape + (2,), dtype=np.uint8)
        self.dataset = self._load_dataset()

    def reset(self, seed):
        """
        Important: the observation must be a numpy array
        :return: (np.array) 
        """
        super().reset(seed = seed)
        # randomly select an image as reference
        self._ref_img = self.dataset[self.np_random.integers(len(self.dataset))]
        # reset the canvas as a white empty one
        self._canvas = torch.full(self.img_shape, 255)
        # restart from 0th step
        self._step_cnt = 0
        return self._get_obs()

    def step(self, action:np.ndarray):
        # apply action
        vertices = action[:-4].reshape(2,-1)
        rgb = np.round(action[-4:-1] * 255).astype(int)
        alpha = action[-1]
        rr,cc = skdraw.polygon(vertices[1:], vertices[2:])
        # alpha blending: new_color = (alpha)*(foreground_color) + (1 - alpha)*(background_color) 
        # see https://graphics.fandom.com/wiki/Alpha_blending
        self._canvas[:,rr,cc] *= 1 - alpha
        self._canvas[:,rr,cc] += rgb * alpha
        # calculate reward
        reward = structural_similarity_index_measure(self._canvas, self._ref_img,data_range=255)
        # Is max step reached?
        self._step_cnt += 1
        done = (self._step_cnt == self.max_step)
        # Optionally we can pass additional info, we are not using that for now
        return self._get_obs(), reward, done, self._get_info() 

    def render(self):
        raise NotImplementedError()

    def close(self):
        pass

    def _get_obs(self):
        return torch.stack([self._ref_img, self._canvas], dim = -1)

    def _get_info(self):
        return {"step": self._step_cnt}

    def _load_dataset(self):
        transforms = tsfm.Compose([tsfm.Resize(self.img_shape), tsfm.ToTensor()])
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
