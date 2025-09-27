// Compile: g++ -std=c++17 -O3 -pthread -fopenmp hw1.cpp -o hw1
// Execute: srun -A ACD114118 -n1 -c${threads} ./hw1 ${input}

#include <bits/stdc++.h>
using namespace std;

const int MAX_CELLS = 256;
struct State {
    int player;
    bitset<MAX_CELLS> boxes;
};

int rows=0, cols=0;
vector<bool> walls(MAX_CELLS,false), targets(MAX_CELLS,false), fragile(MAX_CELLS,false), dead_zone(MAX_CELLS,false);
vector<pair<int,int>> id_to_rc(MAX_CELLS);
int dr[4] = {-1,1,0,0}, dc[4] = {0,0,-1,1};
char dir_char[4] = {'W','S','A','D'}; // up, down, left, right

const int BEAM_WIDTH = 200000;

// ---------- Utilities ----------
inline string state_key(const State &s) {
    string k;
    k.reserve(rows*cols + 8);
    for (int i = 0; i < rows*cols; ++i) k.push_back(s.boxes[i] ? '1' : '0');
    k.push_back('|');
    k += to_string(s.player);
    return k;
}

State parse_state_key(const string &k) {
    State s;
    int sep = k.find('|');
    string b = k.substr(0, sep);
    int N = rows*cols;
    for (int i = 0; i < N && i < (int)b.size(); ++i) s.boxes[i] = (b[i] == '1');
    s.player = stoi(k.substr(sep+1));
    return s;
}

inline bool in_bounds_idx(int idx) { return idx >=0 && idx < rows*cols; }

// ---------- Deadlock detection ----------
bool is_deadlock_simple(const State &s) {
    int N = rows*cols;
    for (int i=0;i<N;++i){
        if (!s.boxes[i]) continue;
        if (targets[i]) continue;
        if (fragile[i]) continue; // skip fragile boxes for deadlock
        auto [r,c] = id_to_rc[i];
        auto blocked = [&](int rr,int cc)->bool{
            if(rr<0||rr>=rows||cc<0||cc>=cols) return true;
            int idx = rr*cols+cc;
            return walls[idx];
        };
        bool up=blocked(r-1,c), down=blocked(r+1,c), left=blocked(r,c-1), right=blocked(r,c+1);
        if ((up||down)&&(left||right)) return true;
    }
    return false;
}

bool is_deadlock(const State &s) {
    if(is_deadlock_simple(s)) return true;
    int N = rows*cols;
    for(int i=0;i<N;++i){
        if(!s.boxes[i]) continue;
        if(dead_zone[i]) return true;
    }
    return false;
}

// ---------- Player reachability ----------
bitset<MAX_CELLS> player_reachable(int start, const bitset<MAX_CELLS> &boxes) {
    bitset<MAX_CELLS> seen;
    queue<int> q;
    seen[start]=1; q.push(start);
    while(!q.empty()){
        int u = q.front(); q.pop();
        auto [r,c] = id_to_rc[u];
        for(int d=0;d<4;++d){
            int nr=r+dr[d], nc=c+dc[d];
            if(nr<0||nr>=rows||nc<0||nc>=cols) continue;
            int v = nr*cols+nc;
            if(walls[v] || boxes[v]) continue;
            if(!seen[v]) { seen[v]=1; q.push(v); }
        }
    }
    return seen;
}

// ---------- BFS player path ----------
string bfs_player_path(const State &s,int from_idx,int to_idx){
    if(from_idx==to_idx) return string();
    int N = rows*cols;
    vector<int> prev(N,-1), prev_dir(N,-1);
    queue<int> q; vector<char> seen(N,0);
    q.push(from_idx); seen[from_idx]=1;
    bool found=false;
    while(!q.empty() && !found){
        int u=q.front(); q.pop();
        auto [r,c] = id_to_rc[u];
        for(int d=0;d<4;++d){
            int nr=r+dr[d], nc=c+dc[d];
            if(nr<0||nr>=rows||nc<0||nc>=cols) continue;
            int v=nr*cols+nc;
            if(walls[v] || s.boxes[v]) continue;
            if(seen[v]) continue;
            seen[v]=1; prev[v]=u; prev_dir[v]=d;
            if(v==to_idx){ found=true; break; }
            q.push(v);
        }
    }
    if(!found) return string();
    vector<char> steps;
    int cur=to_idx;
    while(cur!=from_idx){
        steps.push_back(dir_char[prev_dir[cur]]);
        cur=prev[cur];
    }
    reverse(steps.begin(), steps.end());
    return string(steps.begin(), steps.end());
}

// ---------- Heuristic ----------
vector<int> manhattan_target_r, manhattan_target_c;
int heuristic_sum(const State &s){
    int h=0;
    int N=rows*cols;
    vector<pair<int,int>> box_positions;
    for(int i=0;i<N;++i) if(s.boxes[i]) box_positions.push_back(id_to_rc[i]);
    for(auto [br,bc]: box_positions){
        int best=INT_MAX;
        for(size_t t=0;t<manhattan_target_r.size();++t){
            int d=abs(br-manhattan_target_r[t])+abs(bc-manhattan_target_c[t]);
            best=min(best,d);
        }
        h+=best;
    }
    return h;
}

// ---------- Reconstruct moves ----------
string reconstruct_full_moves(const unordered_map<string,pair<string,char>> &parent,const string &goal_key){
    vector<string> keys; string cur=goal_key;
    while(true){
        keys.push_back(cur);
        auto it=parent.find(cur);
        if(it==parent.end()) break;
        const string &pk=it->second.first;
        if(pk.empty()) break;
        cur=pk;
    }
    reverse(keys.begin(), keys.end());
    string result;
    for(size_t i=0;i+1<keys.size();++i){
        State ps=parse_state_key(keys[i]);
        State cs=parse_state_key(keys[i+1]);
        int b=-1, t=-1; int N=rows*cols;
        for(int idx=0;idx<N;++idx){
            if(ps.boxes[idx] && !cs.boxes[idx]){ b=idx; break; }
        }
        for(int idx=0;idx<N;++idx){
            if(!ps.boxes[idx] && cs.boxes[idx]){ t=idx; break; }
        }
        if(b==-1 || t==-1){
            char mv=parent.at(keys[i+1]).second;
            if(mv) result.push_back(mv);
            continue;
        }
        auto [br,bc]=id_to_rc[b]; auto [tr,tc]=id_to_rc[t];
        int d=-1;
        for(int dd=0;dd<4;++dd) if(br+dr[dd]==tr && bc+dc[dd]==tc){ d=dd; break; }
        if(d==-1){ char mv=parent.at(keys[i+1]).second; if(mv) result.push_back(mv); continue; }
        int pr=br-dr[d], pc=bc-dc[d]; int pidx=pr*cols+pc;
        string walk=bfs_player_path(ps, ps.player, pidx);
        result+=walk; result.push_back(dir_char[d]);
    }
    return result;
}

// ---------- Beam A* solver ----------
pair<int,string> astar_push_solver(const State &start){
    int N=rows*cols;
    manhattan_target_r.clear(); manhattan_target_c.clear();
    for(int i=0;i<N;++i) if(targets[i]){ manhattan_target_r.push_back(id_to_rc[i].first); manhattan_target_c.push_back(id_to_rc[i].second); }

    struct Node{
        int f,g; State s;
        bool operator<(const Node &o) const{
            if(f!=o.f) return f>o.f; return g<o.g;
        }
    };

    priority_queue<Node> pq;
    unordered_map<string,int> best_g;
    unordered_map<string,pair<string,char>> parent;

    int h0=heuristic_sum(start);
    string sk=state_key(start);
    best_g[sk]=0; parent[sk]={string(),0};
    pq.push(Node{h0,0,start});

    while(!pq.empty()){
        vector<Node> next_level;
        while(!pq.empty()){
            Node cur=pq.top(); pq.pop();
            State s=cur.s; int g=cur.g;
            string key=state_key(s);
            if(best_g[key]!=g) continue;
            bool done=true;
            for(int i=0;i<N;++i) if(s.boxes[i] && !targets[i]){ done=false; break; }
            if(done) return {g,reconstruct_full_moves(parent,key)};

            bitset<MAX_CELLS> reach=player_reachable(s.player, s.boxes);
            for(int b=0;b<N;++b){
                if(!s.boxes[b]) continue;
                auto [br,bc]=id_to_rc[b];
                for(int d=0;d<4;++d){
                    int pr=br-dr[d], pc=bc-dc[d], tr=br+dr[d], tc=bc+dc[d];
                    if(pr<0||pr>=rows||pc<0||pc>=cols) continue;
                    if(tr<0||tr>=rows||tc<0||tc>=cols) continue;
                    int pidx=pr*cols+pc, tidx=tr*cols+tc;
                    if(!reach[pidx]) continue;
                    // only forbid moving box into fragile, walls, other boxes, or dead zone
                    if(walls[tidx] || s.boxes[tidx] || dead_zone[tidx] || fragile[tidx]) continue;
                    State ns=s; ns.boxes[b]=0; ns.boxes[tidx]=1; ns.player=b;
                    if(is_deadlock(ns)) continue;
                    int h=heuristic_sum(ns), ng=g+1;
                    string nk=state_key(ns);
                    auto it=best_g.find(nk);
                    if(it==best_g.end() || ng<it->second){
                        best_g[nk]=ng; parent[nk]={key,dir_char[d]};
                        next_level.push_back(Node{ng+h, ng, ns});
                    }
                }
            }
        }
        sort(next_level.begin(), next_level.end(), [](Node &a, Node &b){ return a.f < b.f; });
        if((int)next_level.size()>BEAM_WIDTH) next_level.resize(BEAM_WIDTH);
        for(auto &n : next_level) pq.push(n);
    }
    return {-1,""};
}

// ---------- Main ----------
int main(int argc,char** argv){
    ios::sync_with_stdio(false); cin.tie(nullptr);
    if(argc<2){ cerr<<"Usage: "<<argv[0]<<" level.txt\n"; return 1; }
    ifstream fin(argv[1]); if(!fin){ cerr<<"Cannot open "<<argv[1]<<"\n"; return 1; }
    vector<string> lines; string line;
    while(getline(fin,line)) lines.push_back(line);
    if(lines.empty()){ cerr<<"Empty file\n"; return 1; }
    rows=(int)lines.size(); cols=(int)lines[0].size();
    if(rows*cols>=MAX_CELLS){ cerr<<"Map too large\n"; return 1; }

    State start; int idx=0;
    for(int r=0;r<rows;++r){
        if((int)lines[r].size()<cols) lines[r].resize(cols,' ');
        for(int c=0;c<cols;++c,++idx){
            id_to_rc[idx]={r,c};
            char ch=lines[r][c];
            walls[idx]=(ch=='#');
            fragile[idx]=(ch=='!'||ch=='@');  // only empty fragile tiles
            targets[idx]=(ch=='.'||ch=='O'||ch=='X');
            start.boxes[idx]=(ch=='x'||ch=='X');
            if(ch=='o'||ch=='O'||ch=='@'||ch=='!') start.player=idx; // player can start on fragile tile
        }
    }

    // Precompute dead zones (corners with no target)
    for(int i=0;i<rows*cols;++i){
        if(walls[i] || targets[i]) continue;
        auto [r,c]=id_to_rc[i];
        bool up=(r==0 || walls[(r-1)*cols+c]), down=(r==rows-1 || walls[(r+1)*cols+c]);
        bool left=(c==0 || walls[r*cols+(c-1)]), right=(c==cols-1 || walls[r*cols+(c+1)]);
        if((up||down)&&(left||right)) dead_zone[i]=true;
    }

    auto res=astar_push_solver(start);
    if(res.first>=0) cout<<res.second<<"\n";
    else cout<<"No solution found by Beam-A*\n";
    return 0;
}
