


# how to check if a module isn't being called; if not, where does the data come from.


import module_io
import module_model
import module_train
import module_evaluate
import module_diagnostics
import module_report


# read input data
data_input = module_io.read_data()

# select model
model = module_io.read_model()

# train model
model = module_train(data_input, model)

# evaluate model
data_evaluation = module_evaluate(model, data_input)

# read data out of evaluation
data_evaluation = module_io.read_data()

# create diagnostic object
data_diagnostic = module_diagnostic.Diagnostics(data_input, data_evaluation, model)
data_diagnostic.actual = data_evaulation

