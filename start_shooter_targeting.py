from subprocess import PIPE, Popen
import select
import time

brightnesses = []

# run program 10 times and record brightness each time
for i in range(10):
    p = Popen(['python', 'shooter_targeting.py', '5000'], stdout=PIPE)
    for line in iter(p.stdout.readline, b''):
        print line,
        if 'BRIGHTNESS' in line:
            brightnesses.append(float(line.split(':')[-1]))
            print('Brightness ' + str(i))
            break
    p.kill()
    p.wait()


brightnesses.sort()
print(brightnesses)

# continue trying to open until we find something within range of the average of the min 3 brightnesses
target = (brightnesses[0] + brightnesses[1] + brightnesses[2] + brightnesses[3]) / 4
while True:
    p = Popen(['python', 'shooter_targeting.py', '5000'], stdout=PIPE)
    for line in iter(p.stdout.readline, b''):
        if 'BRIGHTNESS' in line:
            brightness = float(line.split(':')[-1])
            break

    # adjust this range with tests probably, right now it just aims to be within 10 brightness in either direction
    if brightness > target + 10:
        print('Incorrect run...')
        p.kill()
        p.wait()
    else:
        print('Correct exposure run found')
        break

