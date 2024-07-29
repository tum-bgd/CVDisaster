# Cross View Disaster - Geo Localization Training

Install the requirement.txt

```bash
pip install -r requirement.txt
```

2. Training the model:

    ```bash
    python train_cvdisaster.py
    ```
If you want to train on specific splits provided in the splits folder change the small\_training parameter in the dataclass of the training to the specific split size: 10, 20, 30, 40, 50, 60. Since we use a pre-trained Sample4Geo Model please download the CVUSA weights from the [repository](https://github.com/Skyy93/Sample4Geo)


3. Evaluate the model

    ```bash
    python eval_cvdisaster.py
    ```
Again specify the split you want to evaluate, also change the checkpoint\_start parameter in the dataclass to the path of the trained weights.


