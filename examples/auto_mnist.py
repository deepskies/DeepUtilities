import dsutils as ds

# dsutils in its state rn

train_loader, test_loader = ds.Baselines('mnist')

model = ds.auto.mlp.MLP()
