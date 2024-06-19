import random
height = 5000
width = 5000


f = open("2.ppm", "w")
f.write("P3\n")
f.write(f"{height} {width}\n")
f.write("1\n")
for i in range(height):
  for i in range(width):
    rand = random.randint(0, 10)
    if rand == 1:
      f.write("1 1 1  ")
    else:
      f.write("0 0 0  ")
  f.write("\n")