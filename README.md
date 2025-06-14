# The Art of Description: Evaluating Human vs. AI-Based Textual Annotations of Mid-Air Hand Gestures

This repository contains code, data, and analysis for evaluating human and AI-generated textual annotations of mid-air hand gestures, focusing on hybrid meeting platforms. The project explores how gestures are described, predicted, and classified using both human input and large language models (LLMs).

## Repository Structure

- `data/` — Processed datasets:
  - `descriptions/` — CSV files with gesture descriptions from humans and various OpenAI models.
  - `predictions/` — CSV files with gesture predictions from humans and models.
- `dataset/` — Raw and reference datasets:
  - `elicit_cam.csv`, `elicit_cam_ns.csv` — Original elicitation data.
  - `ground_truth_commands.csv` — Reference commands for gestures.
  - `images/` — (Not included due to GDPR; see `images/README.md`.)
  - `jsons/` — Per-participant JSON data.
- `outputs/` — Analysis results and statistics:
  - Accuracy and similarity results, ANOVA, chi-square, and Tukey HSD outputs.
- `pgf/` — Plots and visualizations (PGF format for LaTeX).
- `src/` — Python scripts for data processing and analysis:
  - `s0_human_structured_descriptions.py` — Processes human-structured descriptions.
  - `s1_human_non_structured_descriptions.py` — Processes human non-structured descriptions.
  - `s2_openai_structured_descriptions.py`, etc. — Scripts for OpenAI and other model-based descriptions and predictions.
  - `s5_generate_command_predictions.py` — Generates predictions from descriptions.
- `data_analysis.ipynb` — Main Jupyter notebook for statistical analysis and visualization.
- `test.py` — Test script for validating code or data.
- `pyproject.toml`, `poetry.lock`, `environment.yml` — Dependency management files.
- `LICENSE` — License information.

## Data Overview

- **Descriptions:** Textual annotations of gestures, both human- and AI-generated, for various meeting commands.
- **Predictions:** Model and human predictions of commands based on gesture descriptions.
- **Raw Data:** Original participant data and ground truth commands.

## Analysis Workflow

1. **Data Loading & Preprocessing:** All datasets are loaded and preprocessed in `data_analysis.ipynb` and scripts in `src/`.
2. **Description & Prediction Analysis:** Compare human and AI-generated descriptions and predictions.
3. **Statistical Analysis:** Compute accuracy, similarity, and perform statistical tests (ANOVA, chi-square, Tukey HSD).
4. **Visualization:** Generate plots for accuracy, similarity, and other metrics (see `pgf/`).

## Requirements & Installation

- Python >= 3.10

Install dependencies with Poetry or Conda:

```bash
conda update conda
conda env create --file=./environment.yml
conda activate icmi2025
poetry install --no-root
```

## Usage

- Run data processing and analysis scripts in `src/` as needed.
- For full analysis and visualization, open and run `data_analysis.ipynb`.

## Outputs

- Processed datasets and analysis results in `outputs/`.
- Visualizations in `pgf/` (for LaTeX inclusion).

## Citation

If you use this code or data, please cite the associated paper:

> Anonymous Authors (2025). The Art of Description: Evaluating Human vs. AI-Based Textual Annotations of Mid-Air Hand Gestures. In: Proceedings of the 27th ACM International Conference on Multimodal Interaction, ICMI 2025, Canberra, Australia, October 13-17. 2025. https://doi.org/00.0000/000-0-000-00000-0_0

## License

MIT License. See `LICENSE` for details.

## Contact

For questions or collaborations, please contact the authors or refer to the associated publication.

---

**Authors:** Anonymous Authors.
