# UR-DMU Model Inference

This repository contains pretrained UR-DMU models with varying levels of sparsity (70%, 80%, 90%, 95%, and 99%) for inference tasks. Follow the instructions below to set up and run the inference script.

## Requirements

Before running the code, ensure that you have all the required dependencies installed:

```bash
pip install -r requirements.txt
```

## Running the Inference

To perform inference using the pretrained models, run the `ucf_infer.py` script:

```bash
python ucf_infer.py
```

### Choosing a Model

By default, the script uses the model with 99% sparsity. To use a different model, update the following line in the `ucf_infer.py` file:

```python
valid(net, config, test_loader, model_file=os.path.join(args.model_path, "target_99/model_round_4.pkl"))  # Change model here
```

Replace `target_99` with one of the following options to use a model with a different sparsity level:
- `target_70` for 70% sparsity
- `target_80` for 80% sparsity
- `target_90` for 90% sparsity
- `target_95` for 95% sparsity
- `target_99` for 99% sparsity

## Output

The script will output the following information:
- **Model Accuracy**: The accuracy of the selected model on the test dataset.
- **Inference Time**: The time it took to perform inference.

## Training Logs

In the `scc-logs` directory, you can view our training logs, which provide details about the training process for each model.

## Training Your Own Pruned UR-DMU Model

If you want to train your own pruned UR-DMU model, you can run the `ucf_main.py` file:

```bash
python ucf_main.py
```

To adjust the pruning parameters, modify the following line in `ucf_main.py` (line 42):

```python
pruner = ModelPruner(net, total_prune_rounds=5, target_sparsity=0.8)
```

- **`target_sparsity`**: Set this value to your desired sparsity level (e.g., `0.7` for 70% sparsity).
- **`total_prune_rounds`**: Change this value to adjust the number of pruning iterations.

## Example

Here is an example of running the script with the default model:

```bash
python ucf_infer.py
```

Output:
```
Average inference time (ms): 2.61
AUC: 0.5199927223191341
AP: 0.08250531790988762
```

To use a different model, update the `model_file` path as described above, then rerun the script.

---