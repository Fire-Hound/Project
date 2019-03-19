import subprocess

for i in range(100):
    subprocess.run(["raspistill", "-o", "{}.jpg".format(i)])
