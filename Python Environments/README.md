## Environment Install Instructions

To set up the Python environment for this project, follow the instructions below:

Make sure you have the yaml file in the same directory as this README. Then, run the following command in your terminal:

```bash
conda env create -f environment.yaml
``` 


To update and export the environment, use the following commands:

```bash
conda env update -f environment.yaml --prune
conda env export > environment.yaml
```