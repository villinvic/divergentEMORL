import ast
import json


with open('stats_compare/demorl', 'r') as f:
    x = ast.literal_eval(f.read())
print(x)