import os
import sys

# Add root directory to sys.path to enable imports of server, gradcam, etc.
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from server import app

# Vercel Serverless Function entry point
app = app
