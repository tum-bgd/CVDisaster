# Cross View Disaster

## Env

1. prepare docker container

    ```bash
    docker run -it --gpus device=1 --name gcvit --mount type=bind,source="$(pwd)",target=/root tensorflow/tensorflow:2.10.1-gpu
    ```

2. install `gcvit` inside the container

    ```bash
    python -m pip install --upgrade pip
    python -m pip install gcvit tensorflow_addons geojson rasterio scikit-learn --root-user-action=ignore  # anyway, we are in docker ;)
    ```

3. go to directory

    ```bash
    cd ~
    ```

## Steps

1. Run `0_prepare_satellite.py` to generate corresponding satellite images according to SVI locations (lonlat). Satellite tiles will be generated in `./data/01_Satellite/STL_ThreeCategories`

2. `1_svi.py` and `1_sat.py` will generate single-view results (`GCViTTiny`, 5/5 split).

3. `2_cv.py` will generate cross-view results (`GCViTTiny`). You can also change backbones of the model and specify training ratio using command-line options.

  ```bash
  python 2_cv.py --tr-ratio 0.5 --backbone tiny
  ```

4. `3_estimation.py` can create a text file where each line contains the estimation results, ground truth and corresponding images (ID).


## References

- Hatamizadeh, A., Yin, H., Heinrich, G., Kautz, J., & Molchanov, P. (2023, July). Global context vision transformers. In International Conference on Machine Learning (pp. 12633-12646). PMLR.

- https://github.com/awsaf49/gcvit-tf

