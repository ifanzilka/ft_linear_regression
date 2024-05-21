import pandas as pd
from typing import Union
import inspect

def read_dataframe_file(path_to_file:str) -> Union[pd.DataFrame,None]:
    if path_to_file.endswith('csv'):
        return pd.read_csv(path_to_file)
    elif path_to_file.endswith('pickle'):
        return pd.read_pickle(path_to_file)
    else:
        return None
    

def get_funcions_str(base_functions):
    function_strings = [inspect.getsource(f).strip() for f in base_functions]
    concatenated = "\n".join(function_strings)

    return concatenated
