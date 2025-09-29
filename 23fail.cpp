// Optimization Goals:
// 1. Critical Speedup: Incremental Player Reachability (Calculate BFS once per node, not once per move).
// 2. OpenMP Fix: Correct parallel syntax errors.
// 3. Keep Fine-Grained Locking and Hungarian Heuristic.

// Compile: g++ -std=c++17 -O3 -pthread -fopenmp hw1.cpp -o hw1
// Execute: srun -A ACD114118 -n1 -c${threads} ./hw1 ${input}

#include <bits/stdc++.h>
#include <omp.h>
#include <mutex>
using namespace std;

const int MAX_CELLS = 256;
const int INF = 1000000007;

struct State {
    int player = -1; 
    bitset<MAX_CELLS> boxes;
};

// --- START: State Hash and Locking (Kept for multi-threading) ---
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

constexpr int NUM_LOCKS = 256; 
std::mutex state_locks[NUM_LOCKS];

inline int get_lock_index(const State& s) {
    size_t h = StateHash{}(s); 
    return h & (NUM_LOCKS - 1); 
}
// --- END: State Hash and Locking ---

int rows=0, cols=0;
// dead_zone is now only used for pre-computation if enabled, but checked in is_deadlock
vector<bool> walls(MAX_CELLS,false), targets(MAX_CELLS,false), fragile(MAX_CELLS,false), dead_zone(MAX_CELLS,false); 
vector<pair<int,int>> id_to_rc(MAX_CELLS);
int dr[4] = {-1,1,0,0}, dc[4] = {0,0,-1,1};
char dir_char[4] = {'W','S','A','D'}; 

// ---------- Deadlock detection (Conservative/Dynamic Checks Only) ----------

// is_deadlock_simple: Corner check
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

// is_deadlock: Corner + Dynamic Perimeter Check + dead_zone pre-check (if enabled)
bool is_deadlock(const State &s) {
    if(is_deadlock_simple(s)) return true;
    int N = rows*cols;
    
    for(int i=0;i<N;++i){
        if(!s.boxes[i]) continue;
        if(targets[i]) continue;
        
        // dead_zone acts as a weak initial filter.
        if(dead_zone[i]) return true; 
        
        // --- Dynamic Perimeter Deadlock Check ---
        auto [r,c] = id_to_rc[i];

        auto is_wall_blocked = [&](int rr, int cc) -> bool {
            if (rr < 0 || rr >= rows || cc < 0 || cc >= cols) return true;
            return walls[rr * cols + cc];
        };
        
        // Check if blocked by ANY obstacle (wall or box)
        auto is_blocked_any = [&](int rr, int cc) -> bool {
            if (rr < 0 || rr >= rows || cc < 0 || cc >= cols) return true;
            int idx = rr * cols + cc;
            return walls[idx] || s.boxes[idx];
        };

        // Trapped horizontally by walls AND blocked vertically by ANY obstacle
        bool wall_block_h = is_wall_blocked(r, c-1) && is_wall_blocked(r, c+1);
        if (wall_block_h) {
            bool blocked_any_v = is_blocked_any(r-1, c) && is_blocked_any(r+1, c);
            if (blocked_any_v) return true;
        }

        // Trapped vertically by walls AND blocked horizontally by ANY obstacle
        bool wall_block_v = is_wall_blocked(r-1, c) && is_wall_blocked(r+1, c);
        if (wall_block_v) {
            bool blocked_any_h = is_blocked_any(r, c-1) && is_blocked_any(r, c+1);
            if (blocked_any_h) return true;
        }
    }
    return false;
}

// ---------- Player reachability (Unchanged, but now called less frequently) ----------
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

// BFS player path (Unchanged)
string bfs_player_path(const State &s,int from_idx,int to_idx){
    // ... (omitted for brevity, assume correct from previous version)
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


// ---------- Heuristic (Unchanged) ----------
vector<int> manhattan_target_r, manhattan_target_c;
int heuristic_sum(const State &s){
    // ... (omitted for brevity, assume correct)
    int N=rows*cols;
    vector<pair<int,int>> box_positions;
    box_positions.reserve(manhattan_target_r.size()); 

    for(int i=0;i<N;++i) if(s.boxes[i]) box_positions.push_back(id_to_rc[i]);
    int n = box_positions.size();
    if (n == 0) return 0;

    vector<vector<int>> A(n+1, vector<int>(n+1, 0));
    for(int i=0; i<n; ++i){
        auto [br, bc] = box_positions[i];
        for(int j=0; j<n; ++j){
            int d = abs(br - manhattan_target_r[j]) + abs(bc - manhattan_target_c[j]);
            A[i+1][j+1] = d;
        }
    }

    vector<int> u(n+1), v(n+1), p(n+1), way(n+1);
    for (int i=1; i<=n; ++i) {
        p[0] = i;
        int j0 = 0;
        vector<int> minv(n+1, INF);
        vector<bool> used(n+1, false);
        do {
            used[j0] = true;
            int i0 = p[j0], delta = INF, j1;
            for (int j=1; j<=n; ++j)
                if (!used[j]) {
                    int cur = A[i0][j] - u[i0] - v[j];
                    if (cur < minv[j])
                        minv[j] = cur, way[j] = j0;
                    if (minv[j] < delta)
                        delta = minv[j], j1 = j;
                }
            for (int j=0; j<=n; ++j)
                if (used[j])
                    u[p[j]] += delta, v[j] -= delta;
                else
                    minv[j] -= delta;
            j0 = j1;
        } while (p[j0] != 0);
        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }
    return -v[0];
}

// ---------- Reconstruct moves (Unchanged) ----------
string reconstruct_full_moves(const unordered_map<State, pair<State, char>, StateHash, StateEqual> &parent, const State &goal){
    // ... (omitted for brevity, assume correct)
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

// ---------- Beam A* solver (CRITICAL OPTIMIZATION: Incremental Reachability) ----------
pair<int,string> astar_push_solver(const State &start, int beam_width, long long initial_branching){
    int N=rows*cols;
    // ... (manhattan target setup)
    manhattan_target_r.clear(); manhattan_target_c.clear();
    for(int i=0;i<N;++i) if(targets[i]){ manhattan_target_r.push_back(id_to_rc[i].first); manhattan_target_c.push_back(id_to_rc[i].second); }

    struct Node{
        int f,g; State s;
        bool operator<(const Node &o) const{
            if(f!=o.f) return f>o.f; return g<o.g;
        }
    };

    priority_queue<Node> pq;
    unordered_map<State,int,StateHash,StateEqual> best_g;
    unordered_map<State,pair<State,char>,StateHash,StateEqual> parent; 

    int h0=heuristic_sum(start);
    
    best_g[start]=0; 
    parent[start] = {State{-1, bitset<MAX_CELLS>{}}, 0}; 
    
    pq.push(Node{h0,0,start});

    long long last_raw_per_node = 0;

    while(!pq.empty()){
        vector<Node> current_level;
        while(!pq.empty()){
            current_level.push_back(pq.top());
            pq.pop();
        }
        
        long long est_branching = last_raw_per_node > 0 ? last_raw_per_node : initial_branching;
        long long estimated_candidates = (long long)current_level.size() * est_branching;
        bool use_deferred = estimated_candidates <= (long long)beam_width * 5LL;

        vector<Node> next_level_candidates;
        long long raw_generated = 0;

        if (use_deferred) {
            vector<vector<pair<Node, pair<State, char>>>> thread_candidates(omp_get_max_threads());

            // FIX: Removed duplicate 'dc' from shared clause
            #pragma omp parallel for default(none) shared(current_level, N, best_g, thread_candidates, targets, id_to_rc, dr, dc, rows, cols, walls, fragile, dead_zone, dir_char) 
            for(size_t i=0; i<current_level.size(); ++i){
                Node cur=current_level[i]; 
                State s=cur.s; int g=cur.g;
                int thread_id = omp_get_thread_num();

                auto it_g = best_g.find(s);
                if(it_g==best_g.end() || it_g->second != g) continue; 
                
                bool done=true;
                for(int j=0;j<N;++j) if(s.boxes[j] && !targets[j]){ done=false; break; }
                if(done) continue; 
                
                // CRITICAL OPTIMIZATION: Compute Reachability ONCE per node
                bitset<MAX_CELLS> reach=player_reachable(s.player, s.boxes); 

                for(int b=0;b<N;++b){
                    if(!s.boxes[b]) continue;
                    auto [br,bc]=id_to_rc[b];
                    for(int d=0;d<4;++d){
                        int pr=br-dr[d], pc=bc-dc[d], tr=br+dr[d], tc=bc+dc[d];
                        if(pr<0||pr>=rows||pc<0||pc>=cols) continue;
                        if(tr<0||tr>=rows||tc<0||tc>=cols) continue;
                        int pidx=pr*cols+pc, tidx=tr*cols+tc;
                        
                        // Use the pre-computed reachability map
                        if(!reach[pidx]) continue; 
                        
                        if(walls[tidx] || s.boxes[tidx] || fragile[tidx] || dead_zone[tidx]) continue;
                        
                        State ns=s; ns.boxes[b]=0; ns.boxes[tidx]=1; ns.player=b;
                        
                        // Removed the overly aggressive 'is_player_trapped' check.

                        if(is_deadlock(ns)) continue;
                        int h=heuristic_sum(ns), ng=g+1;
                        
                        thread_candidates[thread_id].push_back({Node{ng+h, ng, ns}, {s, dir_char[d]}});
                    }
                }
            } 

            // ... (Sequential map update block - Unchanged)
            vector<pair<Node, pair<State, char>>> candidates;
            for(auto &vec : thread_candidates){
                candidates.insert(candidates.end(), vec.begin(), vec.end());
            }
            raw_generated = candidates.size();

            vector<Node> pruned_next;
            pruned_next.reserve(candidates.size());
            for(auto &cand : candidates) {
                const Node &n = cand.first;
                const State &ns = n.s;
                int ng = n.g;
                auto it = best_g.find(ns);
                if(it == best_g.end() || ng < it->second){
                    best_g[ns] = ng; 
                    parent[ns] = cand.second;
                    pruned_next.push_back(n);
                }
            }
            next_level_candidates = std::move(pruned_next);
            
        } else {
            // High branching (Parallel map update with Fine-Grained Locking)
            vector<vector<Node>> thread_next_level(omp_get_max_threads());
            vector<long long> thread_raw(omp_get_max_threads(), 0);

            #pragma omp parallel for default(none) shared(current_level, N, best_g, parent, thread_next_level, thread_raw, targets, id_to_rc, dr, dc, rows, cols, walls, fragile, dead_zone, dir_char, state_locks) 
            for(size_t i=0; i<current_level.size(); ++i){
                Node cur=current_level[i]; 
                State s=cur.s; int g=cur.g;
                int thread_id = omp_get_thread_num();

                auto it_g = best_g.find(s);
                if(it_g==best_g.end() || it_g->second != g) continue; 
                
                bool done=true;
                for(int j=0;j<N;++j) if(s.boxes[j] && !targets[j]){ done=false; break; }
                if(done) continue; 

                // CRITICAL OPTIMIZATION: Compute Reachability ONCE per node
                bitset<MAX_CELLS> reach=player_reachable(s.player, s.boxes);
                
                for(int b=0;b<N;++b){
                    if(!s.boxes[b]) continue;
                    auto [br,bc]=id_to_rc[b];
                    for(int d=0;d<4;++d){
                        int pr=br-dr[d], pc=bc-dc[d], tr=br+dr[d], tc=bc+dc[d];
                        if(pr<0||pr>=rows||pc<0||pc>=cols) continue;
                        if(tr<0||tr>=rows||tc<0||tc>=cols) continue;
                        int pidx=pr*cols+pc, tidx=tr*cols+tc;
                        
                        // Use the pre-computed reachability map
                        if(!reach[pidx]) continue;
                        
                        if(walls[tidx] || s.boxes[tidx] || fragile[tidx] || dead_zone[tidx]) continue;
                        
                        State ns=s; ns.boxes[b]=0; ns.boxes[tidx]=1; ns.player=b;
                        
                        // Removed the overly aggressive 'is_player_trapped' check.

                        if(is_deadlock(ns)) continue;
                        int h=heuristic_sum(ns), ng=g+1;
                        
                        thread_raw[thread_id]++;

                        int lock_idx_ns = get_lock_index(ns);
                        std::lock_guard<std::mutex> lock_ns(state_locks[lock_idx_ns]);
                        
                        auto it=best_g.find(ns);
                        if(it==best_g.end() || ng<it->second){
                            best_g[ns]=ng; 
                            parent[ns]={s,dir_char[d]}; 
                            thread_next_level[thread_id].push_back(Node{ng+h, ng, ns});
                        }
                    }
                }
            } 

            // ... (Parallel map update block - Unchanged)
            for(auto &vec : thread_next_level){
                next_level_candidates.insert(next_level_candidates.end(), vec.begin(), vec.end());
            }
            for(auto r : thread_raw) {
                raw_generated += r;
            }
        }
        // ... (Beam pruning - Unchanged)
        if (current_level.size() > 0) {
            last_raw_per_node = raw_generated / current_level.size();
        }

        size_t select_size = std::min((size_t)beam_width, next_level_candidates.size());
        std::partial_sort(next_level_candidates.begin(), next_level_candidates.begin() + select_size, next_level_candidates.end(), [](const Node &a, const Node &b){ return a.f < b.f; });
        next_level_candidates.resize(select_size);
        
        for(auto &n : next_level_candidates) pq.push(n);

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

// ---------- Main (Reverted to simple pre-computation for safety) ----------
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

    int N = rows * cols;
    
    // --- Reverted Static Deadlock Precomputation ---
    // Only pre-mark simple 1x1 corners as dead_zone for a minimal filter.
    fill(dead_zone.begin(), dead_zone.end(), false);

    auto is_blocked = [&](int r, int c) -> bool {
        if (r < 0 || r >= rows || c < 0 || c >= cols) return true; 
        int idx = r * cols + c;
        return walls[idx]; 
    };
    
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int i00 = r * cols + c;
            if (walls[i00] || targets[i00]) continue;

            // Simple 1x1 Corner Check
            bool up_b = is_blocked(r - 1, c), down_b = is_blocked(r + 1, c);
            bool left_b = is_blocked(r, c - 1), right_b = is_blocked(r, c + 1);
            if ((up_b || down_b) && (left_b || right_b)) {
                dead_zone[i00] = true;
            }
        }
    }
    // --- End Reverted Deadlock Precomputation ---
    
    // Compute initial valid pushes as proxy for initial_branching
    bitset<MAX_CELLS> reach = player_reachable(start.player, start.boxes);
    long long initial_branching = 0;
    for(int b = 0; b < N; ++b) {
        if(!start.boxes[b]) continue;
        for(int d = 0; d < 4; ++d) {
            int pr = id_to_rc[b].first - dr[d], pc = id_to_rc[b].second - dc[d];
            int tr = id_to_rc[b].first + dr[d], tc = id_to_rc[b].second + dc[d];
            if(pr < 0 || pr >= rows || pc < 0 || pc >= cols) continue;
            if(tr < 0 || tr >= rows || tc < 0 || tc >= cols) continue;
            int pidx = pr * cols + pc, tidx = tr * cols + tc;
            if(!reach[pidx]) continue;
            if(walls[tidx] || start.boxes[tidx] || fragile[tidx] || dead_zone[tidx]) continue;
            State ns = start; ns.boxes[b] = 0; ns.boxes[tidx] = 1; ns.player = b;
            
            if(is_deadlock(ns)) continue;
            ++initial_branching;
        }
    }
    if(initial_branching == 0) initial_branching = 1;

    // Try increasing beam widths
    vector<int> beam_widths = {500, 5000, 30000, 60000, 100000, 200000};
    for(int bw : beam_widths){
        auto res=astar_push_solver(start, bw, initial_branching);
        if(res.first>=0){
            cout<<res.second<<"\n";
            return 0;
        }
    }
    cout<<"No solution found by Beam-A*\n";
    return 0;
}