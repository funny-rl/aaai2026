
n=int(input())
a=[int(j) for j in input().split()]
dp=[[0]*(n+1) for i in range(n+1)]

for i in range(n)[::-1]:
    for j in range(i,n):
        if (n+i-j-1)%2==0:
            dp[i][j]=max(dp[i+1][j]+a[i],dp[i][j-1]+a[j])
        else:
            dp[i][j]=min(dp[i+1][j]-a[i],dp[i][j-1]-a[j])
print(dp[0][n-1])

