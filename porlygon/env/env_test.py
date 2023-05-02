import unittest
import numpy as np
from porlygon.env import DrawPolygonEnv


class TestDrawPolygonEnvStep(unittest.TestCase):
    def setUp(self):
        self.env = DrawPolygonEnv()
        self.env.reset()
        self.action_shape = self.env.action_space.shape
        self.env_x = self.env.img_shape[1]
        self.env_y = self.env.img_shape[2]

    def action_same_color_polygon(self):
        action = np.zeros(self.action_shape)
        action[-4:-1] = self.env._canvas[0, 0, 0] / 255
        return action

    def action_red_rectangle(self):
        action = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1])
        return action

    def action_black_triangle(self):
        action = np.array([0, 0, 0.5, 1, 0, 1, 0.5, 0, 0, 0, 0, 1])
        return action

    def action_random(self):
        return self.env.action_space.sample()

    def point_at(self, x: float, y: float):
        return int(x * self.env_x), int(y * self.env_y)

    def test_constraints(self):
        self.env.reset()
        action = self.action_random()
        new_obs, reward, done, _, _ = self.env.step(action)

        self.assertTrue(isinstance(new_obs, dict), "Observation should be a dictionary")
        self.assertTrue(isinstance(reward, float), "Reward should be a float")
        self.assertTrue(isinstance(done, bool), "Done should be a boolean")

        for key in ["reference", "canvas"]:
            self.assertTrue(key in new_obs, f"Observation should have a '{key}' key")
            self.assertTrue(
                new_obs[key].shape == self.env.observation_space[key].shape,
                f"Shape of '{key}' should match observation_space",
            )
            self.assertTrue(
                new_obs[key].dtype == self.env.observation_space[key].dtype,
                f"Dtype of '{key}' should match observation_space",
            )

    def test_same_color_different_alpha_polygons(self):
        for alpha in [0.0, 0.5, 1.0]:
            with self.subTest(alpha=alpha):
                self.env.reset()
                orig_canvas = self.env._canvas.copy()

                action = self.action_same_color_polygon()
                action[-1] = alpha

                new_obs, _, _, _, _ = self.env.step(action)
                new_canvas = new_obs["canvas"]

                self.assertTrue(
                    np.all(orig_canvas == new_canvas),
                    f"Canvas changed for alpha {alpha}",
                )

    def test_blend_large_red_rectangle(self):
        self.env.reset()
        orig_canvas = self.env._canvas.copy()

        action = self.action_red_rectangle()
        new_obs, _, _, _, _ = self.env.step(action)
        new_canvas = new_obs["canvas"]

        # Check if the blending works as expected for a single point
        blended_point = new_canvas[:, 50, 50]
        expected_blended_color = (
            (orig_canvas[:, 50, 50] * (1 - action[-1]))
            + (action[-4:-1] * 255 * action[-1])
        ).astype(int)
        self.assertTrue(
            np.all(blended_point == expected_blended_color),
            "Blending did not work as expected, "
            f"expect {expected_blended_color}, get {blended_point}",
        )

    def test_large_triangle_shape(self):
        self.env.reset()

        action = self.action_black_triangle()
        new_obs, _, _, _, _ = self.env.step(action)
        new_canvas = new_obs["canvas"]

        inside_point = self.point_at(0.25, 0.25)
        outside_point = self.point_at(0.75, 0.75)

        # Check if a point inside the triangle has a different color
        self.assertFalse(
            np.any(new_canvas[:, inside_point[0], inside_point[1]] != 0),
            "Inside point color did not changed/wrong",
        )

        # Check if a point outside the triangle still has the original color
        self.assertTrue(
            np.all(new_canvas[:, outside_point[0], outside_point[1]] == 255),
            "Outside point color changed",
        )

    def test_reward_and_termination(self):
        self.env.reset()
        action = self.action_red_rectangle()
        _, reward_before, _, _, _ = self.env.step(action)

        for _ in range(self.env.max_step - 2):
            _, reward, done, _, _ = self.env.step(action)
            self.assertFalse(done, "Environment terminated early")
            self.assertGreaterEqual(reward, reward_before, "Reward should not decrease")
            reward_before = reward

        _, _, done, _, _ = self.env.step(action)
        self.assertTrue(
            done, "Environment should terminate after reaching the max_step"
        )


if __name__ == "__main__":
    unittest.main()
