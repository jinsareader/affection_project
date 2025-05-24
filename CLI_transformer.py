import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "common"))
from common.CLIform import Mainform

Mainform("korean_vector.pkl","transformer.onnx",30)