#include<iostream>
#include<vector>
#include<cmath>
#include<string>
#include <algorithm>
#include <queue>
#include <unordered_set>
#include <unordered_map>
using namespace std;
typedef long long ll;
#define MOD 1000000007

int main()
{
    ll tc;
    scanf("%d", &tc);
    for(ll i=1; i<= tc; i++)
    {
        ll n;
        vector<ll> set;
        scanf("%lld", &n);
        ll sum = pow(2,30)-1;
        for(ll i=1; i<30; i++)
        {
            printf(" ");
            printf("%lld",pow(2,i-1));
        }
        for(int i=0; i<70; i++)
        {
            printf(" ");
            printf("%lld", pow(10,9)-i);
        }
        for(ll i=0; i<n; i++)
        {
            ll inp;
            scanf("%lld",inp);
            set.push_back(inp);
            sum += inp;
        }
    }
}