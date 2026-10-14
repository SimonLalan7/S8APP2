import math

i = 13
k = 3
p = (k - 1) // 2
s = 1
out = math.floor((i + 2 * p - k) / s ) + 1
print(out)

RF_in = 39
jump_in=8
stride=1

RF_out = RF_in + (k - 1) * jump_in
jump_out = jump_in * stride
print(RF_out, jump_out)