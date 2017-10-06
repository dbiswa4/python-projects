n = 45

DP = [0 for k in range(n + 1)]
DP[0] = 1
DP[1] = 1
DP[2] = 1
DP[3] = 2

for i in range(4, n+1):
    DP[i] = DP[i-1] + DP[i-3] + DP[i-4]

print DP[n]