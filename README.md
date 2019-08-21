# dsutils
Deep Skies Utilities for IO, Pre-Processing, Training, and Diagnostics

# [IO notebook](https://colab.research.google.com/drive/1qw73O-zC8_3Tmlq-1rpdLjZ_omr0u0Xj)
# no link [pre processing]()
# no link [training/model building]()
# no link [plotting/diagnostics]()`

end goal:
```
import dsutils as ds

data = ds.get_dataset('strong_lens', scale=False)  # have option for scaling

# run baseline models, save model and plots
ds.baselines(data, save=True, plots=True)
```

Adding the option to pass a package ambiguous architecture as well as an actual model instantiation would be good to have.


For gcs interaction:
  [Setting up authentication](https://cloud.google.com/storage/docs/reference/libraries)
Maybe source solution from here:
https://github.com/deepskies/DataPreparation/blob/master/Strong_Lens_No_Lens_Source_Demo.ipynb


While the GCS upload functions have not been written, I am assuming that all datasets are on cloud or accessible via pytorch/tf datasets.

h5 stuff:

https://gist.github.com/branislav1991/2c4a225e3066cf1542dacdbc11f51e09
https://gist.github.com/branislav1991/4c143394bdad612883d148e0617bdccd
