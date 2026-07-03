import AOLEN

datasets = ['LED_gradual', 'LED_abrupt', 'RBFBlips', 'sea', 'tree', 'electricity', 'covertype', 'weather', 'hyperplane',
            'kddcup99']
dataflow_settings = {
    'TTime': 1000,
    'k': 500,
    'beta': 0.99,
    'train': 1000,
}

model_settings = {
    'hidden_size': 100,
    'lr': 0.01,
}
#random seed
# randomfactor=42
if __name__ == '__main__':
    for dataset in datasets:
        AOLEN.run_experiments(dataset, dataflow_settings, model_settings,randomfactor=None)