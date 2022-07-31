#include<iostream>
#include<vector>
#include<cmath>
#include<string>
#include<algorithm>
#include<queue>
#include<unordered_set>
#include<unordered_map>
#include<stack>
#include<deque>
using namespace std;
typedef long long ll;
#define MOD 1000000007

bool check(string s)
{
    for(int i=0; i<s.length()-2; i++)
    {
        if(s.substr(i+2).find(s[i])!=-1 && s[i+1]-s[i]!=0) return true;
    }
    return false;
}

void solve()
{
    int n,k;
    cin >> n >> k;
    vector<ll> list;
    for(int i=0; i<n; i++)
    {
        ll inp;
        cin >> inp;
        list.push_back(inp);
    }
    ll sum_a = 0;
    ll sum_a2 = 0;
    for(int i=0; i<n; i++)
    {
        sum_a += list[i];
        sum_a2 += list[i] * list[i];
    }
    if(k==1)
    {
        if(sum_a ==0 && (sum_a2 - sum_a * sum_a) !=0)
        {
            cout << "IMPOSSIBLE" << endl;
            return;
        }
        else if(sum_a ==0)
        {
            cout << "0" << endl;
            return;
        }
        else if((sum_a2-sum_a*sum_a) % (2*sum_a)!=0)
        {
            cout << "IMPOSSIBLE" << endl;
            return;
        }
        else
        {
            cout << (sum_a2-sum_a*sum_a) /(2*sum_a) << endl;
            return;
        }
    }
    else
    {
        if((sum_a2+sum_a*sum_a) % 2 !=0)
        {
            cout << "IMPOSSIBLE" << endl;
        }
        else
        {
            cout << 1-sum_a << " " << (sum_a2+sum_a*sum_a)/2-sum_a << endl;
        }
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