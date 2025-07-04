import subprocess

# Get the list of installed packages using pip
with open('requirements.txt', 'w') as f:
    subprocess.run(['pip', 'freeze'], stdout=f)

print("requirements.txt has been generated.")
