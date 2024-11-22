# NumberPlateRecognition

This projects aims to create a program to make images and videos safe to upload under strict data protection laws.

# Developer Information

## Rules

- Always create a new brach for a new feature. Use the first letter of your name as the start of the branch name (e.g. c-topic)
- Use typing for all functions and classes

## Conda Environment

All the dependencies for this project are saved in the environment.yml file for a conda environment. Use 
```
conda env create --name NumberPlateRecognition --file=environment.yml
``` 
to create the conda environment. Use `ctrl+shift+p` and search for `python:Select Interpreter` to select the environment as interpreter in vs code.

When installing new packages create a new file with:
```
conda env export > environment.yml
``` 
To update your conda environment with the new file use:
```
conda env update --file environment.yml --prune
```

## Resources

### Datasets
- [PapersWithCode](https://paperswithcode.com/datasets?task=object-detection&mod=images&page=1)
