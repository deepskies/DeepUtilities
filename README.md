# dsutils
Deep Skies Utilities for IO, Pre-Processing, Training, and Diagnostics

```
import dsutils as ds

data = ds.get_dataset('strong_lens', scale=False)  # have option for scaling

# run baseline models, save model and plots
ds.baselines(data, save=True, plots=True)

```

