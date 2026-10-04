# Changelog

All notable changes to this project are documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `movie` preset: locked camera with one static crop per shot and hard cuts.
- `--start` / `--duration` to reframe only a window of the source.
- `--fast`, `--imgsz`, `--analyze-stride`, `--face-stride`, `--device`, and `--saliency-model off`.
- `--no-two-person-framing` to switch off two-person framing in presets that enable it.

### Changed
- `sports` preset follows the ball and the nearest player, with a dead zone, motion lead, and two-person framing on by default; `max_zoom` and step limits are lower.
- Follow presets hard-cut to the subject on scene changes instead of easing from frame center.

## [0.1.0] - 2026-04-19

### Added
- Initial release of the Auto Vertical Reframe CLI.
- Scene-aware vertical reframing pipeline built on YOLOv11 segmentation, MediaPipe face/pose, and PySceneDetect.
- Presets: `talking_head`, `sports`, `pets`, `cars`.
- Handcrafted and DeepGaze-MR saliency backends with CPU/CUDA/MPS selection.
- macOS `run_verthor.command` launcher with native dialogs.
- Debug preview export and JSON summary logging.
