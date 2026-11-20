from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data import download
if __name__=='__main__': download(Path(__file__).resolve().parents[1])
