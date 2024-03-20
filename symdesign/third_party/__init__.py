from pathlib import Path
import sys

third_party_dir = Path(__file__).absolute().parent
sys.path.extend([str(third_party_dir)])
