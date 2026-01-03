# 🐳 Deploy_YOLO_Docker

[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![YOLOv8](https://img.shields.io/badge/Model-YOLOv11--ONNX-00ff00)](https://docs.ultralytics.com/)
[![License: AGPL 3.0](https://img.shields.io/badge/License-AGPL_v3.0-blue.svg)](https://opensource.org/licenses/AGPL-3.0)

## 📝 Overview

This project provides a robust, containerized environment to deploy **YOLO** (You Only Look Once) models for real-time object detection. By leveraging **Docker** and **ONNX Runtime**, it ensures a seamless "Build Once, Run Anywhere" workflow, eliminating the common "it works on my machine" issues in Computer Vision pipelines.

## ✨ Key Features

* **Containerized Inference:** Isolated environment with all CV2 and ONNX dependencies pre-configured.
* **Production Ready:** Optimized for high-performance inference using the `.onnx` format.
* **Orchestration Friendly:** Full support for `docker-compose` for multi-service scaling.
* **Flexible Configuration:** Dynamic model loading via environment variables.

## 📁 Directory Structure

```text
Deploy_YOLO_Docker/
├── models/           # Pre-trained YOLO models (.onnx)
├── src/              # Python source code (Inference scripts)
├── Dockerfile        # Container definition
├── pyproject.toml    # Python dependencies
└── README.md         # Documentation
```

## 🚀 Getting Started

### Prerequisites

* [Docker](https://www.docker.com/get-started/) installed.

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/CarlosM1024/Deploy_YOLO_Docker.git
    ```

2. Navigate to the project directory:

    ```bash
    cd Deploy_YOLO_Docker
    ```

3. Build the image:

    ```bash
    docker build -t yolo-app .
    ```

## 💻 Usage

1. Running the Container. Map the internal API port to your host machine:

    ```bash
    docker run -p 8080:8080 yolo-app
    ```

## 📄 License

This project is licensed under the GNU Affero General Public License v3.0. See the LICENSE file for details.

## 🙏 Credits and Acknowledgments

### Technical Foundation

This deployment architecture is built upon the robust standards provided by **Ultralytics** for enterprise-grade computer vision. Specifically, the containerization strategy was informed by their official guide:

* **Source:** [Vertex AI Deployment with Docker](https://docs.ultralytics.com/guides/vertex-ai-deployment-with-docker/)
* **Organization:** [Ultralytics](https://ultralytics.com/)

### Educational Resources & Ecosystem

The successful implementation of this pipeline was made possible through the documentation and community insights from:

* **Core Libraries:** Official documentation of [YOLO](https://www.ultralytics.com/), [ONNX Runtime](https://onnxruntime.ai/), and [FastApi](https://fastapi.tiangolo.com/).
* **Containerization:** Docker official best practices for Python-based Microservices.
* **Community:** Technical discussions and optimization tips from the GitHub community.

### Technical Statement

While this project follows the architectural patterns established by Ultralytics, **all code within this repository has been independently implemented, tested, and optimized by me.**

## 🤝 Contributing

If you'd like to contribute to this project, feel free to submit a pull request. Please make sure your code follows the existing style and includes appropriate comments.

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes.
4. Push to the branch.
5. Submit a pull request.

## 👤 Author

````Carlos Antonio Martinez Miranda````

GitHub: [@CarlosM1024](https://github.com/CarlosM1024)
