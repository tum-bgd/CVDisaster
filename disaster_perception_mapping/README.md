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

## Results

Backbone model is `GCViT-Tiny`.

### Single-view vs. Cross-view

50/50 train/validation

<table class="tg"><thead>
  <tr>
    <th class="tg-0pky" rowspan="2"></th>
    <th class="tg-fymr" colspan="2">SVI</th>
    <th class="tg-fymr" colspan="2">Satellite</th>
    <th class="tg-fymr" colspan="2">CrossView</th>
  </tr>
  <tr>
    <th class="tg-0pky">P</th>
    <th class="tg-0pky">R</th>
    <th class="tg-0pky">P</th>
    <th class="tg-0pky">R</th>
    <th class="tg-0pky">P</th>
    <th class="tg-0pky">R</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-dvpl">Minor</td>
    <td class="tg-0pky">0.7861</td>
    <td class="tg-0pky">0.8815</td>
    <td class="tg-0pky">0.7252</td>
    <td class="tg-0pky">0.8237</td>
    <td class="tg-0pky">0.8264</td>
    <td class="tg-0pky">0.8764</td>
  </tr>
  <tr>
    <td class="tg-dvpl">Moderate</td>
    <td class="tg-0pky">0.6204</td>
    <td class="tg-0pky">0.5630</td>
    <td class="tg-0pky">0.5569</td>
    <td class="tg-0pky">0.3908</td>
    <td class="tg-0pky">0.6429</td>
    <td class="tg-0pky">0.6681</td>
  </tr>
  <tr>
    <td class="tg-dvpl">Severe</td>
    <td class="tg-0pky">0.8188</td>
    <td class="tg-0pky">0.7219</td>
    <td class="tg-0pky">0.6580</td>
    <td class="tg-0pky">0.7515</td>
    <td class="tg-0pky">0.8915</td>
    <td class="tg-0pky">0.7188</td>
  </tr>
  <tr>
    <td class="tg-6ic8"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">Overall Acc.</td>
    <td class="tg-0pky" colspan="2">0.7450</td>
    <td class="tg-0pky" colspan="2">0.6707</td>
    <td class="tg-0pky" colspan="2">0.7796</td>
  </tr>
  <tr>
    <td class="tg-0pky">Overall Prec.</td>
    <td class="tg-0pky" colspan="2">0.7417</td>
    <td class="tg-0pky" colspan="2">0.6467</td>
    <td class="tg-0pky" colspan="2">0.7869</td>
  </tr>
  <tr>
    <td class="tg-0pky">Overall Reca.</td>
    <td class="tg-0pky" colspan="2">0.7221</td>
    <td class="tg-0pky" colspan="2">0.6569</td>
    <td class="tg-0pky" colspan="2">0.7544</td>
  </tr>
</tbody></table>

### Training ratio

<table class="tg"><thead>
  <tr>
    <th class="tg-c3ow"></th>
    <th class="tg-c3ow" colspan="2">1:9</th>
    <th class="tg-c3ow" colspan="2">2:8</th>
    <th class="tg-c3ow" colspan="2">3:7</th>
    <th class="tg-baqh" colspan="2">4:6</th>
    <th class="tg-baqh" colspan="2">5:5</th>
    <th class="tg-baqh" colspan="2">6:4</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow">P</td>
    <td class="tg-c3ow">R</td>
    <td class="tg-c3ow">P</td>
    <td class="tg-c3ow">R</td>
    <td class="tg-c3ow">P</td>
    <td class="tg-c3ow">R</td>
    <td class="tg-baqh">P</td>
    <td class="tg-baqh">R</td>
    <td class="tg-baqh">P</td>
    <td class="tg-baqh">R</td>
    <td class="tg-baqh">P</td>
    <td class="tg-baqh">R</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Minor</td>
    <td class="tg-c3ow">0.7735</td>
    <td class="tg-c3ow">0.7710</td>
    <td class="tg-c3ow">0.7963</td>
    <td class="tg-c3ow">0.8527</td>
    <td class="tg-c3ow">0.8180</td>
    <td class="tg-c3ow">0.7787</td>
    <td class="tg-baqh">0.8305</td>
    <td class="tg-baqh">0.8147</td>
    <td class="tg-baqh">0.8264</td>
    <td class="tg-baqh">0.8764</td>
    <td class="tg-baqh">0.7703</td>
    <td class="tg-baqh">0.8289</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Moderate</td>
    <td class="tg-c3ow">0.5113</td>
    <td class="tg-c3ow">0.5230</td>
    <td class="tg-c3ow">0.6133</td>
    <td class="tg-c3ow">0.5766</td>
    <td class="tg-c3ow">0.5405</td>
    <td class="tg-c3ow">0.5684</td>
    <td class="tg-baqh">0.5784</td>
    <td class="tg-baqh">0.6367</td>
    <td class="tg-baqh">0.6429</td>
    <td class="tg-baqh">0.6681</td>
    <td class="tg-baqh">0.5938</td>
    <td class="tg-baqh">0.5561</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Severe</td>
    <td class="tg-c3ow">0.6848</td>
    <td class="tg-c3ow">0.6667</td>
    <td class="tg-c3ow">0.7804</td>
    <td class="tg-c3ow">0.7343</td>
    <td class="tg-c3ow">0.7075</td>
    <td class="tg-c3ow">0.7247</td>
    <td class="tg-baqh">0.8000</td>
    <td class="tg-baqh">0.7220</td>
    <td class="tg-baqh">0.8915</td>
    <td class="tg-baqh">0.7188</td>
    <td class="tg-baqh">0.7656</td>
    <td class="tg-baqh">0.7259</td>
  </tr>
  <tr>
    <td class="tg-7btt"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-baqh"></td>
    <td class="tg-baqh"></td>
    <td class="tg-baqh"></td>
    <td class="tg-baqh"></td>
    <td class="tg-baqh"></td>
    <td class="tg-baqh"></td>
  </tr>
  <tr>
    <td class="tg-c3ow">Overall Acc.</td>
    <td class="tg-c3ow" colspan="2">0.6684</td>
    <td class="tg-c3ow" colspan="2">0.7380</td>
    <td class="tg-c3ow" colspan="2">0.7005</td>
    <td class="tg-baqh" colspan="2">0.7389</td>
    <td class="tg-baqh" colspan="2">0.7796</td>
    <td class="tg-baqh" colspan="2">0.7131</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Overall Prec.</td>
    <td class="tg-c3ow" colspan="2">0.6565</td>
    <td class="tg-c3ow" colspan="2">0.7300</td>
    <td class="tg-c3ow" colspan="2">0.6887</td>
    <td class="tg-baqh" colspan="2">0.7363</td>
    <td class="tg-baqh" colspan="2">0.7869</td>
    <td class="tg-baqh" colspan="2">0.7099</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Overall Reca.</td>
    <td class="tg-c3ow" colspan="2">0.6536</td>
    <td class="tg-c3ow" colspan="2">0.7212</td>
    <td class="tg-c3ow" colspan="2">0.6906</td>
    <td class="tg-baqh" colspan="2">0.7245</td>
    <td class="tg-baqh" colspan="2">0.7544</td>
    <td class="tg-baqh" colspan="2">0.7036</td>
  </tr>
</tbody></table>

## References

- Hatamizadeh, A., Yin, H., Heinrich, G., Kautz, J., & Molchanov, P. (2023, July). Global context vision transformers. In International Conference on Machine Learning (pp. 12633-12646). PMLR.

- https://github.com/awsaf49/gcvit-tf

