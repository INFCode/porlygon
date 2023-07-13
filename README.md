# Porlygon

Porlygon is a project focused on reconstructing images with semi-transparent polygons using a reinforcement learning approach. The aim of this project is to explore various reinforcement learning algorithms, gain experience in loss design, and build environments for image reconstruction. The project is written in Python.

## Installation

To get started with Porlygon, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/INFCode/porlygon
   ```

2. Install the required dependencies using Poetry. If you don't have Poetry installed, run the following command:
   ```
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. Navigate to the project directory:
   ```
   cd porlygon
   ```

4. Install the core system:
   ```
   poetry build
   ```

   Alternatively, if you want to include the support of a human-friendly window display, use the following command instead:
   ```
   cd porlygon
   poetry build --all-extra
   ```

## Usage

Porlygon is currently under development and not yet usable. The intended functionality is to provide a user interface that allows users to interact with trained models for image reconstruction. The details of the user interface are still being designed and will be implemented in future iterations.

## Model and Training Code

Porlygon utilizes the following libraries for reinforcement learning:

- `gym`: A library for defining reinforcement learning environments.
- `gymnasium`: A library for creating custom environments. (Note: Planned transition from OpenAI Gym to Farama-Foundation's Gymnasium.)

The model architecture involves the use of Convolutional Neural Networks (CNNs) for image and action processing, as well as Multi-Layer Perceptrons (MLPs) for generating the results. The project leverages the `tianshou` reinforcement learning library based on PyTorch.

## Future Goals

Future plans for Porlygon include exploring additional reinforcement learning algorithms and expanding the user interface capabilities for users to interact with trained models.

## Contributing

Contributions to Porlygon are welcome! If you encounter any issues or have feature requests, please submit them through GitHub issues. Although handling pull requests from others is a relatively new experience, contributions through pull requests are also appreciated. Let's collaborate and improve Porlygon together!

## License

This project is licensed under the [MIT License](LICENSE.txt).
