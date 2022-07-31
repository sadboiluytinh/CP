#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef vector<ll> vll;
typedef pair<ll, ll> pll;
#define MOD 1000000007

void solve()
{
   ll n,l;
   cin >> n >> l;

   ll posdi[n][2];
   for(ll i=0; i<n; i++)
   {
       cin >> posdi[i][0] >> posdi[i][1];
   }
}

signed main()
{
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    int tc;
    cin >> tc;
    for(int i=1; i<=tc; i++)
    {
        cout<<"Case #"<<i<<":"<<" ";
        solve();
        cout << endl;
    }
}