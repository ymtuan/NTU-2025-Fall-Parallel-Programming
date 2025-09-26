// Compile: g++ -std=c++17 -O3 -pthread -fopenmp hw1.cpp -o hw1
// Execute: srun -A ACD114118 -n1 -c${threads} ./hw1 ${input}

#include <bits/stdc++.h>
using namespace std;

const int MAX_CELLS = 256;
struct State {
    int player;
    bitset<MAX_CELLS> boxes;
};

int rows = 0, cols = 0;
vector<bool> walls(MAX_CELLS,false), targets(MAX_CELLS,false), fragile(MAX_CELLS,false);
vector<pair<int,int>> id_to_rc(MAX_CELLS);
int dr[4] = {-1,1,0,0}, dc[4] = {0,0,-1,1};
char dir_char[4] = {'W','S','A','D'}; // up, down, left, right

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

// helper: whether a cell index is valid
inline bool in_bounds_idx(int idx) {
    return idx >= 0 && idx < rows*cols;
}

// ---------- Deadlock: simple corner (treat fragile as blocking for boxes) ----------
bool is_deadlock_simple(const State &s) {
    int N = rows * cols;
    for (int i = 0; i < N; ++i) {
        if (!s.boxes[i]) continue;
        if (targets[i]) continue;
        auto [r,c] = id_to_rc[i];
        // For boxes, a neighbor is blocking if it's a wall or a fragile tile (boxes can't occupy fragile)
        auto blocked = [&](int rr, int cc)->bool{
            if (rr < 0 || rr >= rows || cc < 0 || cc >= cols) return true;
            int idx = rr*cols + cc;
            return walls[idx] || fragile[idx];
        };
        bool up = blocked(r-1,c);
        bool down = blocked(r+1,c);
        bool left = blocked(r,c-1);
        bool right = blocked(r,c+1);
        if ((up || down) && (left || right)) return true;
    }
    return false;
}

// ---------- Player reachability (BFS) ----------
// Player can step on fragile tiles; boxes block movement.
bitset<MAX_CELLS> player_reachable(int start, const bitset<MAX_CELLS> &boxes) {
    bitset<MAX_CELLS> seen;
    queue<int> q;
    seen[start] = 1;
    q.push(start);
    while(!q.empty()){
        int u = q.front(); q.pop();
        auto [r,c] = id_to_rc[u];
        for (int d=0; d<4; ++d) {
            int nr=r+dr[d], nc=c+dc[d];
            if (nr<0||nr>=rows||nc<0||nc>=cols) continue;
            int v = nr*cols + nc;
            if (walls[v]) continue;      // walls block
            if (boxes[v]) continue;      // boxes block player walking
            if (!seen[v]) { seen[v]=1; q.push(v); }
        }
    }
    return seen;
}

// ---------- BFS path (player-only) returning move letters ----------
string bfs_player_path(const State &s, int from_idx, int to_idx) {
    if (from_idx == to_idx) return string();
    int N = rows*cols;
    vector<int> prev(N, -1), prev_dir(N, -1);
    queue<int> q;
    vector<char> seen(N,0);
    q.push(from_idx); seen[from_idx]=1;
    bool found=false;
    while(!q.empty() && !found){
        int u = q.front(); q.pop();
        auto [r,c] = id_to_rc[u];
        for (int d=0; d<4; ++d) {
            int nr=r+dr[d], nc=c+dc[d];
            if (nr<0||nr>=rows||nc<0||nc>=cols) continue;
            int v = nr*cols + nc;
            if (walls[v] || s.boxes[v]) continue;
            if (seen[v]) continue;
            seen[v]=1; prev[v]=u; prev_dir[v]=d;
            if (v == to_idx) { found=true; break; }
            q.push(v);
        }
    }
    if (!found) return string(); // shouldn't happen if reachability checked
    vector<char> steps;
    int cur = to_idx;
    while (cur != from_idx) {
        int d = prev_dir[cur];
        steps.push_back(dir_char[d]);
        cur = prev[cur];
    }
    reverse(steps.begin(), steps.end());
    return string(steps.begin(), steps.end());
}

// ---------- Precompute distances to nearest target for boxes ----------
// Boxes cannot be on fragile tiles, so do BFS on cells that are not walls and not fragile.
vector<int> dist_to_target;
void compute_dist_to_targets() {
    int N = rows*cols;
    dist_to_target.assign(N, INT_MAX);
    queue<int> q;
    for (int i=0;i<N;++i) {
        if (targets[i] && !fragile[i] && !walls[i]) { // target must be box-walkable
            dist_to_target[i] = 0;
            q.push(i);
        }
    }
    while(!q.empty()){
        int u = q.front(); q.pop();
        auto [r,c] = id_to_rc[u];
        for (int d=0; d<4; ++d) {
            int nr=r+dr[d], nc=c+dc[d];
            if (nr<0||nr>=rows||nc<0||nc>=cols) continue;
            int v = nr*cols + nc;
            if (walls[v] || fragile[v]) continue; // boxes can't pass here
            if (dist_to_target[v] > dist_to_target[u] + 1) {
                dist_to_target[v] = dist_to_target[u] + 1;
                q.push(v);
            }
        }
    }
}

// ---------- Heuristic: sum of distances for boxes to nearest target ----------
int heuristic_sum(const State &s) {
    int h = 0;
    int N = rows*cols;
    for (int i=0;i<N;++i) if (s.boxes[i]) {
        // if box is on fragile (shouldn't happen for valid input) treat as dead
        if (fragile[i]) return INT_MAX/4;
        int d = dist_to_target[i];
        if (d==INT_MAX) return INT_MAX/4;
        h += d;
    }
    return h;
}

// ---------- Reconstruct full move string (walks + push) ----------
string reconstruct_full_moves(const unordered_map<string, pair<string,char>> &parent, const string &goal_key) {
    vector<string> keys;
    string cur = goal_key;
    while (true) {
        keys.push_back(cur);
        auto it = parent.find(cur);
        if (it == parent.end()) break;
        const string &pk = it->second.first;
        if (pk.empty()) break;
        cur = pk;
    }
    reverse(keys.begin(), keys.end()); // start -> ... -> goal

    string result;
    for (size_t i = 0; i + 1 < keys.size(); ++i) {
        State ps = parse_state_key(keys[i]);
        State cs = parse_state_key(keys[i+1]);
        int b = -1, t = -1;
        int N = rows*cols;
        for (int idx=0; idx<N; ++idx) {
            if (ps.boxes[idx] && !cs.boxes[idx]) { b = idx; break; }
        }
        for (int idx=0; idx<N; ++idx) {
            if (!ps.boxes[idx] && cs.boxes[idx]) { t = idx; break; }
        }
        if (b == -1 || t == -1) {
            char mv = parent.at(keys[i+1]).second;
            if (mv) result.push_back(mv);
            continue;
        }
        auto [br, bc] = id_to_rc[b];
        auto [tr, tc] = id_to_rc[t];
        int d = -1;
        for (int dd=0; dd<4; ++dd) if (br + dr[dd] == tr && bc + dc[dd] == tc) { d = dd; break; }
        if (d == -1) {
            char mv = parent.at(keys[i+1]).second;
            if (mv) result.push_back(mv);
            continue;
        }
        int pr = br - dr[d], pc = bc - dc[d];
        int pidx = pr*cols + pc;
        // BFS walk on ps (player can step on fragile!)
        string walk = bfs_player_path(ps, ps.player, pidx);
        result += walk;
        result.push_back(dir_char[d]);
    }
    return result;
}

// ---------- A* push-based solver (optimal) ----------
pair<int,string> astar_push_solver(const State &start) {
    int N = rows*cols;
    compute_dist_to_targets();
    for (int i=0;i<N;++i) if (start.boxes[i]) {
        if (fragile[i]) return {-1,""}; // invalid (boxes on fragile) — assignment says input valid though
        if (dist_to_target[i] == INT_MAX) return {-1,""};
    }

    struct Node {
        int f;
        int g;
        State s;
        bool operator<(const Node &o) const {
            if (f != o.f) return f > o.f; // min-heap
            return g < o.g;
        }
    };

    priority_queue<Node> pq;
    unordered_map<string,int> best_g;
    unordered_map<string, pair<string,char>> parent; // child -> (parent_key, push_char)

    int h0 = heuristic_sum(start);
    if (h0 >= INT_MAX/4) return {-1,""};
    string sk = state_key(start);
    best_g[sk] = 0;
    parent[sk] = make_pair(string(), (char)0);
    pq.push(Node{h0, 0, start});

    while (!pq.empty()) {
        Node cur = pq.top(); pq.pop();
        State s = cur.s;
        int g = cur.g;
        string key = state_key(s);
        auto itg = best_g.find(key);
        if (itg != best_g.end() && g != itg->second) continue;

        // goal test
        bool done = true;
        for (int i=0;i<N;++i) if (s.boxes[i] && !targets[i]) { done = false; break; }
        if (done) {
            string full = reconstruct_full_moves(parent, key);
            return {g, full};
        }

        // reachable player places (player can step on fragile)
        bitset<MAX_CELLS> reach = player_reachable(s.player, s.boxes);

        // generate pushes
        for (int b=0;b<N;++b) {
            if (!s.boxes[b]) continue;
            auto [br, bc] = id_to_rc[b];
            for (int d=0; d<4; ++d) {
                int pr = br - dr[d], pc = bc - dc[d];
                int tr = br + dr[d], tc = bc + dc[d];
                if (pr<0||pr>=rows||pc<0||pc>=cols) continue;
                if (tr<0||tr>=rows||tc<0||tc>=cols) continue;
                int pidx = pr*cols + pc;
                int tidx = tr*cols + tc;
                if (!reach[pidx]) continue;
                // target cell must be free AND box-walkable (not wall, not fragile)
                if (walls[tidx] || s.boxes[tidx] || fragile[tidx]) continue;
                State ns = s;
                ns.boxes[b] = 0; ns.boxes[tidx] = 1; ns.player = b;
                if (is_deadlock_simple(ns)) continue;
                int h = heuristic_sum(ns);
                if (h >= INT_MAX/4) continue;
                int ng = g + 1;
                string nk = state_key(ns);
                auto it = best_g.find(nk);
                if (it == best_g.end() || ng < it->second) {
                    best_g[nk] = ng;
                    parent[nk] = make_pair(key, dir_char[d]);
                    pq.push(Node{ng + h, ng, ns});
                }
            }
        }
    }
    return {-1,""};
}

// ---------- Main ----------
int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (argc < 2) { cerr<<"Usage: "<<argv[0]<<" level.txt\n"; return 1; }
    ifstream fin(argv[1]);
    if (!fin) { cerr<<"Cannot open "<<argv[1]<<"\n"; return 1; }

    vector<string> lines; string line;
    while (getline(fin, line)) lines.push_back(line);
    if (lines.empty()) { cerr<<"Empty file\n"; return 1; }

    rows = (int)lines.size();
    cols = (int)lines[0].size();
    if (rows*cols >= MAX_CELLS) { cerr<<"Map too large\n"; return 1; }

    State start; int idx = 0;
    for (int r=0; r<rows; ++r) {
        if ((int)lines[r].size() < cols) lines[r].resize(cols, ' ');
        for (int c=0; c<cols; ++c, ++idx) {
            id_to_rc[idx] = {r,c};
            char ch = lines[r][c];
            walls[idx] = (ch == '#');
            fragile[idx] = (ch == '@' || ch == '!');
            targets[idx] = (ch == '.' || ch == 'O' || ch == 'X');
            start.boxes[idx] = (ch == 'x' || ch == 'X');
            if (ch == 'o' || ch == 'O' || ch == '!') start.player = idx;
        }
    }

    auto t0 = chrono::high_resolution_clock::now();
    auto res = astar_push_solver(start);
    auto t1 = chrono::high_resolution_clock::now();
    double elapsed = chrono::duration_cast<chrono::duration<double>>(t1 - t0).count();

    if (res.first >= 0) {
        cout << "Solution found with pushes = " << res.first << "\n";
        cout << "Full moves (WASD): " << res.second << "\n";
    } else {
        cout << "No solution found by A*\n";
    }
    cout << "Elapsed time: " << elapsed << " seconds\n";
    return 0;
}
