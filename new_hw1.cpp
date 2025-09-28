// Grok, some parallel, 25 TLE+

// Compile: g++ -std=c++17 -O3 -pthread -fopenmp hw1.cpp -o hw1
// Execute: srun -A ACD114118 -n1 -c${threads} ./hw1 ${input}

#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

const int MAX_CELLS = 256;
struct State {
    int player;
    bitset<MAX_CELLS> boxes;
};

// --- START OPTIMIZATION: Custom Hash and Equality for State ---
struct StateHash {
    size_t operator()(const State& s) const {
        size_t h = std::hash<int>{}(s.player);
        
        const unsigned long long* data = (const unsigned long long*)&s.boxes;
        constexpr int N_WORDS = MAX_CELLS / (sizeof(unsigned long long) * 8); 
        
        for (int i = 0; i < N_WORDS; ++i) {
            h ^= std::hash<unsigned long long>{}(data[i]) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};

struct StateEqual {
    bool operator()(const State& a, const State& b) const {
        return a.player == b.player && a.boxes == b.boxes;
    }
};
// --- END OPTIMIZATION ---

int rows=0, cols=0;
vector<bool> walls(MAX_CELLS,false), targets(MAX_CELLS,false), fragile(MAX_CELLS,false), dead_zone(MAX_CELLS,false);
vector<pair<int,int>> id_to_rc(MAX_CELLS);
int dr[4] = {-1,1,0,0}, dc[4] = {0,0,-1,1};
char dir_char[4] = {'W','S','A','D'}; // up, down, left, right

// ---------- Utilities (removed state_key and parse_state_key) ----------

inline bool in_bounds_idx(int idx) { return idx >=0 && idx < rows*cols; }

// ---------- Deadlock detection (MODIFIED) ----------

// is_deadlock_simple is the original corner check
bool is_deadlock_simple(const State &s) {
    int N = rows*cols;
    for (int i=0;i<N;++i){
        if (!s.boxes[i]) continue;
        if (targets[i]) continue;
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

// is_deadlock uses Static Deadlocks + the new, safer Perimeter Check
bool is_deadlock(const State &s) {
    if(is_deadlock_simple(s)) return true;
    int N = rows*cols;
    
    for(int i=0;i<N;++i){
        if(!s.boxes[i]) continue;
        if(targets[i]) continue;
        
        // Static Dead Zones (original corners + 2x2 precomputed)
        if(dead_zone[i]) return true;
        
        // --- ADDED: Balanced Perimeter Deadlock Check ---
        auto [r,c] = id_to_rc[i];

        // Hard Blockers: Walls or Map Edges (Cannot be moved)
        auto is_wall_blocked = [&](int rr, int cc) -> bool {
            if (rr < 0 || rr >= rows || cc < 0 || cc >= cols) return true;
            return walls[rr * cols + cc];
        };
        
        // Any Blocker: Walls, Map Edges, OR Boxes (Used for the perpendicular axis)
        auto is_blocked_any = [&](int rr, int cc) -> bool {
            if (rr < 0 || rr >= rows || cc < 0 || cc >= cols) return true;
            int idx = rr * cols + cc;
            return walls[idx] || s.boxes[idx];
        };

        // Condition 1: Trapped horizontally by walls AND blocked vertically by ANY obstacle
        // (The box is in a vertical corridor pinned by walls/edges)
        bool wall_block_h = is_wall_blocked(r, c-1) && is_wall_blocked(r, c+1);
        if (wall_block_h) {
            bool blocked_any_v = is_blocked_any(r-1, c) && is_blocked_any(r+1, c);
            if (blocked_any_v) return true;
        }

        // Condition 2: Trapped vertically by walls AND blocked horizontally by ANY obstacle
        // (The box is in a horizontal corridor pinned by walls/edges)
        bool wall_block_v = is_wall_blocked(r-1, c) && is_wall_blocked(r+1, c);
        if (wall_block_v) {
            bool blocked_any_h = is_blocked_any(r, c-1) && is_blocked_any(r, c+1);
            if (blocked_any_h) return true;
        }
    }
    return false;
}

// ---------- Player reachability & BFS player path (unchanged) ----------
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

// ---------- Heuristic (unchanged) ----------
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

// ---------- Reconstruct moves (MODIFIED) ----------
string reconstruct_full_moves(const unordered_map<State, pair<State, char>, StateHash, StateEqual> &parent, const State &goal){
    vector<State> states;
    State cur = goal;
    while(true){
        states.push_back(cur);
        auto it = parent.find(cur);
        if(it == parent.end()) break;
        State pk = it->second.first;
        if(pk.player == -1) break;
        cur = pk;
    }
    reverse(states.begin(), states.end());
    string result;
    for(size_t i=0;i+1<states.size();++i){
        State ps = states[i];
        State cs = states[i+1];
        int b=-1, t=-1; int N=rows*cols;
        for(int idx=0;idx<N;++idx){
            if(ps.boxes[idx] && !cs.boxes[idx]){ b=idx; break; }
        }
        for(int idx=0;idx<N;++idx){
            if(!ps.boxes[idx] && cs.boxes[idx]){ t=idx; break; }
        }
        if(b==-1 || t==-1){
            auto it = parent.find(cs);
            if(it != parent.end()) {
                char mv = it->second.second;
                if(mv) result.push_back(mv);
            }
            continue;
        }
        auto [br,bc]=id_to_rc[b]; auto [tr,tc]=id_to_rc[t];
        int d=-1;
        for(int dd=0;dd<4;++dd) if(br+dr[dd]==tr && bc+dc[dd]==tc){ d=dd; break; }
        if(d==-1){ 
            auto it = parent.find(cs);
            if(it != parent.end()) {
                char mv = it->second.second;
                if(mv) result.push_back(mv);
            }
            continue; 
        }
        int pr=br-dr[d], pc=bc-dc[d]; int pidx=pr*cols+pc;
        string walk=bfs_player_path(ps, ps.player, pidx);
        result+=walk; result.push_back(dir_char[d]);
    }
    return result;
}

// ---------- Beam A* solver (MODIFIED) ----------
pair<int,string> astar_push_solver(const State &start, int beam_width){
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
    // NOTE: unordered_map access needs careful synchronization.
    // Given the nature of A* (and Beam A*), only writes need protection.
    unordered_map<State,int,StateHash,StateEqual> best_g;
    unordered_map<State,pair<State,char>,StateHash,StateEqual> parent; 

    int h0=heuristic_sum(start);
    
    best_g[start]=0; 
    parent[start] = {State{-1, bitset<MAX_CELLS>{}}, 0};
    
    pq.push(Node{h0,0,start});

    while(!pq.empty()){
        vector<Node> current_level;
        while(!pq.empty()){
            current_level.push_back(pq.top());
            pq.pop();
        }
        
        // Parallelize the expansion of nodes in the current level
        // 'next_level' is now a list of thread-local vectors
        vector<vector<Node>> thread_next_level(omp_get_max_threads());

        // The OpenMP loop runs in parallel
        #pragma omp parallel for default(none) shared(current_level, N, best_g, parent, thread_next_level, beam_width, targets, id_to_rc, dr, dc, rows, cols, walls, fragile, dead_zone, dir_char) 
        for(size_t i=0; i<current_level.size(); ++i){
            Node cur=current_level[i]; 
            State s=cur.s; int g=cur.g;
            int thread_id = omp_get_thread_num();

            // Find current best_g (Read access is okay, but map update must be protected)
            auto it_g = best_g.find(s);
            if(it_g==best_g.end() || it_g->second != g) continue; 
            
            
            // Check for goal state (Read access is safe)
            bool done=true;
            // targets is a shared global variable
            for(int j=0;j<N;++j) if(s.boxes[j] && !targets[j]){ done=false; break; }
            if(done) {
                continue; 
            }

            bitset<MAX_CELLS> reach=player_reachable(s.player, s.boxes);
            for(int b=0;b<N;++b){
                if(!s.boxes[b]) continue;
                // id_to_rc is a shared global variable
                auto [br,bc]=id_to_rc[b];
                for(int d=0;d<4;++d){
                    // dr, dc are shared global variables
                    int pr=br-dr[d], pc=bc-dc[d], tr=br+dr[d], tc=bc+dc[d];
                    // rows, cols are shared global variables
                    if(pr<0||pr>=rows||pc<0||pc>=cols) continue;
                    if(tr<0||tr>=rows||tc<0||tc>=cols) continue;
                    int pidx=pr*cols+pc, tidx=tr*cols+tc;
                    if(!reach[pidx]) continue;
                    // walls, fragile, dead_zone are shared global variables
                    if(walls[tidx] || s.boxes[tidx] || fragile[tidx] || dead_zone[tidx]) continue;
                    State ns=s; ns.boxes[b]=0; ns.boxes[tidx]=1; ns.player=b;
                    if(is_deadlock(ns)) continue;
                    int h=heuristic_sum(ns), ng=g+1;
                    
                    // CRITICAL SECTION: Write access to shared maps
                    #pragma omp critical
                    {
                        auto it=best_g.find(ns);
                        if(it==best_g.end() || ng<it->second){
                            best_g[ns]=ng; 
                            parent[ns]={s,dir_char[d]}; 
                            // Add to thread-local list
                            thread_next_level[thread_id].push_back(Node{ng+h, ng, ns});
                        }
                    } // End Critical Section
                }
            }
        } // End OpenMP parallel for

        // Merge thread-local results into a single vector
        vector<Node> next_level;
        for(auto &vec : thread_next_level){
            next_level.insert(next_level.end(), vec.begin(), vec.end());
        }

        // Sort and select the beam width (Sequential part)
        sort(next_level.begin(), next_level.end(), [](Node &a, Node &b){ return a.f < b.f; });
        if((int)next_level.size()>beam_width) next_level.resize(beam_width);
        
        // Add to priority queue for the next iteration (Sequential part)
        for(auto &n : next_level) pq.push(n);

        // Check the current level for the goal state again (as we skipped immediate returns in the parallel loop)
        for (const auto& cur : current_level) {
            State s = cur.s;
            bool done = true;
            for(int j=0; j<N; ++j) if(s.boxes[j] && !targets[j]){ done=false; break; }
            if(done) {
                return {cur.g, reconstruct_full_moves(parent, s)};
            }
        }

    }
    return {-1,""};
}

// ---------- Main (MODIFIED) ----------
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
            fragile[idx]=(ch=='@'||ch=='!');
            targets[idx]=(ch=='.'||ch=='O'||ch=='X');
            start.boxes[idx]=(ch=='x'||ch=='X');
            if(ch=='o'||ch=='O'||ch=='!') start.player=idx;
        }
    }

    // Precompute dead zones (corners with no target)
    for(int i=0;i<rows*cols;++i){
        if(walls[i] || targets[i]) continue;
        auto [r,c]=id_to_rc[i];
        bool up=(r==0 || walls[(r-1)*cols+c]), down=(r==rows-1 || walls[(r+1)*cols+c]);
        bool left=(c==0 || walls[r*cols+(c-1)]), right=(c==cols-1 || walls[r*cols+(c+1)]);
        
        // Original simple corner check
        if((up||down)&&(left||right)) dead_zone[i]=true;
    }
    
    // --- ADDED: 2x2 Static Deadlock Precomputation ---
    for(int r=1; r<rows-1; ++r) {
        for(int c=1; c<cols-1; ++c) {
            // Check all 4 cells in the 2x2 block
            int i00 = r*cols+c, i10 = (r+1)*cols+c, i01 = r*cols+(c+1), i11 = (r+1)*cols+(c+1);
            
            // Only consider 2x2 blocks of non-target cells
            if (!targets[i00] && !targets[i10] && !targets[i01] && !targets[i11]) {
                
                // Case 1: i00 trapped by Wall Left and Wall Above
                if (walls[i00 - 1] && walls[i00 - cols]) dead_zone[i00]=true;
                
                // Case 2: i01 trapped by Wall Right and Wall Above
                if (walls[i01 + 1] && walls[i01 - cols]) dead_zone[i01]=true;

                // Case 3: i10 trapped by Wall Left and Wall Below
                if (walls[i10 - 1] && walls[i10 + cols]) dead_zone[i10]=true;

                // Case 4: i11 trapped by Wall Right and Wall Below
                if (walls[i11 + 1] && walls[i11 + cols]) dead_zone[i11]=true;
            }
        }
    }
    // --- END ADDED ---
    
    // Try increasing beam widths
    vector<int> beam_widths = {500, 5000, 30000, 60000, 100000, 200000};
    // vector<int> beam_widths = {20000, 50000, 80000, 100000};
    for(int bw : beam_widths){
        auto res=astar_push_solver(start, bw);
        if(res.first>=0){
            cout<<res.second<<"\n";
            // cout<<"Solution found with beam width="<<bw<<"\n";
            return 0;
        }
        // cout<<"Failed with beam width="<<bw<<"\n";
    }
    cout<<"No solution found by Beam-A*\n";
    return 0;
}