# motor-current-pivoting

[![Preprint](https://img.shields.io/badge/engrXiv-10.31224%2F8492-b31b1b.svg)](https://doi.org/10.31224/8492) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23264189.svg)](https://doi.org/10.5281/zenodo.23264189)

Code and data for the preprint **"Gravity Pivoting from Motor Current: Calibration-Free Angle Estimation and the Sensing Budget It Requires"** (Clement Aidoo, CANLABS, 2026), [doi:10.31224/8492](https://doi.org/10.31224/8492).

A parallel-jaw gripper pivots a grasped rod by loosening its grip and letting gravity rotate it. This repository shows, in MuJoCo simulation, that the wrist motor's current is enough to measure and control the rod's angle. It contains:

- a calibration-free estimator: a slow, downhill wrist sweep plus a four-term least-squares fit;
- a model of the motor and current-sensing chain (gear ratio, Coulomb and viscous friction, rotor inertia, ADC resolution, noise);
- a closed-loop pivoting controller that uses only motor current.

**All results are from simulation.** See the paper's Limitations section before reusing any number.

## Repository layout

| File | What it does |
|---|---|
| `mj_env.py` | MuJoCo testbed (soft-finger contacts) and single-trial routine used in Exp A |
| `mj_ctrl.py` | Stateful rig for closed-loop runs (grip ramps, smooth clamp, probe sweep) |
| `motor_model.py` | Motor + current-sensing chain, sweep least-squares estimator, error budget |
| `exp_a.py` | Exp A: 100 randomized objects, saves probe traces to `results/mjA_amp*.npz` |
| `analyze_ab.py` | Exp A analysis and Exp B sensing-budget sweep, writes `results/mj_results.json` |
| `figs_ab.py` | Figures for Exp B |
| `exp_c.py` | Exp C: closed-loop pivoting vs open-loop baselines, writes `results/expC_part*.json` |
| `analyze_c.py` | Exp C analysis and figure, writes `results/expC_results.json` |
| `paper/` | LaTeX source; `gen_numbers.py` generates every number in the paper from `results/` |
| `results/`, `figures/` | The exact outputs reported in the paper |

## Reproduce

Python 3.12. Tested on Linux, CPU only.

```bash
pip install -r requirements.txt

# Exp A: simulate (about 3 min per amplitude)
python exp_a.py 0.075 100
python exp_a.py 0.15 100
python exp_a.py 0.3 100

# Exp A + B analysis (about 5 min) and figures
python analyze_ab.py 0.15
python figs_ab.py

# Exp C: closed loop, 40 objects split over two processes (about 10 min each)
python exp_c.py 0 2 40 &
python exp_c.py 1 2 40 &
wait
python analyze_c.py

# Paper numbers and PDF
cd paper && python gen_numbers.py && pdflatex main.tex && pdflatex main.tex
```

`run_all.sh` runs everything in order. The committed `results/` let you skip the simulations and rerun only the analysis. MuJoCo is deterministic for a given version and platform; on a different platform, simulation outputs may differ in the last digits.

## Notes

- `exp_c.py` also contains `est_from` and `release`, the rolling-window wrist-dither controller described in the paper as a negative result. They are kept for reference and are not used by the final controller.
- An earlier PyBullet version of this testbed, and the audit of a previous study, are documented in a separate technical report and are not part of this repository.

## Citation

If you use this code, please cite the preprint (see `CITATION.cff`). The arXiv identifier will be added once it is announced.

## AI assistance

An AI assistant (Claude, Anthropic) was used to write and debug parts of this code and to draft the paper. The author directed the study, reviewed the code and results, and is responsible for the content.

## Licence

Code: MIT (see `LICENSE`). Data in `results/` and figures: CC BY 4.0.
