import os

def check_direc(str):
    run_cmd(f"mkdir -p {str}")

def run_cmd(str):
    print(str)
    os.system(str)