#/bin/bash -l

git config filter.strip-notebook-output.clean 'jupyter-nbconvert --ClearOutputPreprocessor.enabled=True --to=notebook --stdin --stdout --log-level=ERROR'
