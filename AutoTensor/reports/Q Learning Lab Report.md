The best state found for the iris dataset was:

```python
{
    'layers': [
        {
            'class_name': 'dense',
            'args': {
                'units': 32,
                'activation': 'sigmoid'
            }
        }
    ],
    'compile_args': {
        'loss': 'categorical_crossentropy',
        'optimizer': 'nadam'
    },
    'max_epochs': 200,
    'patience': 1
}
```

This state gave 100% accuracy on the test data set.
