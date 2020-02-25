# DeepUtilities
Deep Skies Utilities for IO, Pre-Processing, Training, and Diagnostics

```
import deeputilities as du

data = du.get_dataset('strong_lens', scale=False)  # have option for scaling

# run baseline models, save model and plots
du.baselines(data, save=True, plots=True)
```

or run `baselines.py` from project parent directory
