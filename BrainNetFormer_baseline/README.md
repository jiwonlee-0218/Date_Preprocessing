# BrainNetFormer

![BrainNetFormer](assets\pics\BrainNetFormer.png)

## Data

Firstly, put the HCP task data in the folder data/ as the following organization:

|--data/

​    |--hcptask_roi-aal.pth

​    |--subtask_labels_detail.npy



**hcptask_roi-aal.pth:** 

- timeseries_list: time series of each subject. Shape: time series length * brain region numbers.

- label_list: label of each subject. Values in ['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM'].

**subtask_labels_detail.npy:** the label of subtasks for different main tasks. For simplicity, we regard the subtask of each main task as the same, since the subtasks of each main task in the HCP task dataset share almost the same pattern. 

## To run the code

```python
python main.py
```

## Model Detail

### Some abbreviations:

b: batch_size;      t: the number of sampling points;     n: the number of brain regions;     w: window size;     l: dynamic_length

(The notations here are a little different from the figure, where the usage of l and t are reversed)

### Input of the model:

- **dyn_t:** Dynamic timeseries. Shape: b t n w.

- **dyn_a:** Dynamic FC matrices. Shape: b t n n.

- **t:** Whole timeseries. Shape: l b n.

- **a:** Static FC matrix. Shape: b n n.

- **sampling_endpoints:** The endpoint of each sampling. List with length == t.

## logs/ & assets/results/

Ablation results are documented in these folders. 

* **sa1:** no spatial attention (fixed at 1 for each node).

* **sa1reg0:** no subtask prediction (reg_subtask == 0).

### Files  

**argv.csv** documents the hyperparameter settings.

**metric.csv** documents the k-fold performance over the main task and subtask. The last four lines record the overall performance over the k-fold (i.e., mean and std values).

## Requirements

- python 3.8.5
- numpy == 1.20.2
- torch == 1.7.0
- torchvision == 0.8.1
- einops == 0.3.0
- sklearn == 0.24.2
- nilearn == 0.7.1
- nipy == 0.5.0
- pingouin == 0.3.11
- tensorboard == 2.5.0
- tqdm == 4.60.0