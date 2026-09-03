# thor_eval

Minimal Thor-side VaVAM-B ego trajectory evaluation.

Files:
- `thor_ego_trajectory_dataset.py`: standalone dataset/token/GT loader
- `min_ade.py`: minimal implementation matching VaVAM `min_ade`
- `12j_vavam_B_evaluate_ego_trajectory.py`: evaluation driver

First test dataset loading only:

```bash
cd ~/vblkdev2/VaVAM_Thor/thor_eval

python3 12j_vavam_B_evaluate_ego_trajectory.py \
  --pickle ../data/nuScenes-mini/nuscenes_mini_data_cleaned.pkl \
  --tokens-root ../data/nuScenes-mini/tokens \
  --dataset-only
```

The TRT adapter in `12j_vavam_B_evaluate_ego_trajectory.py` is intentionally
left as an integration boundary. It should reuse the already validated 12i
GPU-resident prefill/action/Euler implementation rather than introducing a
second TRT runtime implementation.
