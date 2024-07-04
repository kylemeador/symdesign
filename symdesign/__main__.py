import sys
print(sys.path)
# from symdesign.run import app
# Doesn't work with -m/script from
# python -m symdesign.symdesign
from .run import app
# Works with script, doesn't work with -m
# from run import app

app()
