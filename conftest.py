import pytest
import sys
import asyncio

# https://github.com/scikit-learn/scikit-learn/issues/8959
import numpy as np
try:
    np.set_printoptions(sign=' ')
except TypeError:
    pass

'''
Cambio introducido para adaptarse bien al entorno del sistema
operativo Windows 10.
Más información del problema en el siguiente issue:
https://github.com/jupyter/jupyter-sphinx/issues/171#issuecomment-766953182
'''

if sys.version_info[0] == 3 and sys.version_info[1] >= 8\
        and sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

collect_ignore = ['setup.py', 'docs/conf.py']

pytest.register_assert_rewrite("skfda")
