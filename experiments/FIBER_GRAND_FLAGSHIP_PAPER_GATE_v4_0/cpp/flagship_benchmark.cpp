#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <memory>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fg {

using Clock = std::chrono::steady_clock;

static constexpr long double TOL = 1e-18L;

uint64_t low_mask(int bits) {
    if (bits <= 0) return 0ULL;
    if (bits >= 64) return std::numeric_limits<uint64_t>::max();
    return (1ULL << bits) - 1ULL;
}

int bit(uint64_t x, int i) { return int((x >> i) & 1ULL); }

uint64_t delete_bit(uint64_t word, int position, int /*n*/) {
    const uint64_t lower = word & low_mask(position);
    const uint64_t upper = word >> (position + 1);
    return lower | (upper << position);
}

uint64_t insert_bit(uint64_t base, int base_length, int position, int hidden) {
    const uint64_t lower = base & low_mask(position);
    const uint64_t upper = base >> position;
    uint64_t result = lower | (uint64_t(hidden & 1) << position) | (upper << (position + 1));
    if (base_length + 1 < 64) result &= low_mask(base_length + 1);
    return result;
}

long double component_mass(int mismatches, int m, long double p, int streams) {
    if (p == 0.0L) return mismatches == 0 ? 1.0L / streams : 0.0L;
    return std::pow(p, mismatches) * std::pow(1.0L - p, m - mismatches) / streams;
}

long double unnormalized_mass(int mismatches, int m, long double p) {
    if (p == 0.0L) return mismatches == 0 ? 1.0L : 0.0L;
    return std::pow(p, mismatches) * std::pow(1.0L - p, m - mismatches);
}

struct Code {
    std::string family;
    std::string name;
    int n = 0;
    int k = 0;
    int r = 0;
    std::vector<uint64_t> row_masks;
    std::vector<uint32_t> syndrome_columns;

    uint64_t encode(uint64_t message) const {
        uint64_t word = 0;
        for (int j = 0; j < k; ++j) {
            if ((message >> j) & 1ULL) word ^= row_masks[size_t(j)];
        }
        return word;
    }

    uint32_t syndrome(uint64_t word, uint64_t* bitops = nullptr) const {
        uint32_t s = 0;
        uint64_t v = word;
        uint64_t ops = 0;
        while (v) {
            const uint64_t lsb = v & (~v + 1ULL);
            const int pos = __builtin_ctzll(v);
            s ^= syndrome_columns[size_t(pos)];
            v ^= lsb;
            ops += uint64_t(std::max(1, r));
        }
        if (bitops) *bitops += ops;
        return s;
    }

    bool is_codeword(uint64_t word, uint64_t* bitops = nullptr) const {
        return syndrome(word, bitops) == 0;
    }
};

uint64_t crc_remainder(uint64_t message, int k, int r, uint64_t polynomial) {
    __uint128_t value = (__uint128_t(message) << r);
    for (int shift = k - 1; shift >= 0; --shift) {
        if ((value >> (shift + r)) & 1) value ^= (__uint128_t(polynomial) << shift);
    }
    if (r == 64) return uint64_t(value);
    return uint64_t(value) & low_mask(r);
}

std::pair<uint64_t, std::string> crc_polynomial(int r) {
    switch (r) {
        case 3: return {0xB, "CRC-3/GSM-like"};
        case 4: return {0x13, "CRC-4/ITU-like"};
        case 5: return {0x25, "CRC-5/USB-like"};
        case 6: return {0x67, "CRC-6/CDMA2000-A-like"};
        case 7: return {0x89, "CRC-7/MMC-like"};
        case 8: return {0x107, "CRC-8/ATM-like"};
        case 10: return {0x633, "CRC-10/ATM-like"};
        case 12: return {0x180F, "CRC-12/UMTS-like"};
        case 16: return {0x11021, "CRC-16/CCITT-like"};
        default: return {(1ULL << r) | (1ULL << std::max(1, r / 2)) | 1ULL, "deterministic"};
    }
}

Code make_random_linear(int n, int k, std::mt19937_64& rng) {
    Code code;
    code.family = "random_linear";
    code.name = "RLC_n" + std::to_string(n) + "_k" + std::to_string(k);
    code.n = n; code.k = k; code.r = n - k;
    code.row_masks.resize(size_t(k));
    std::uniform_int_distribution<int> coin(0, 1);
    for (int j = 0; j < k; ++j) {
        uint64_t row = 1ULL << j;
        for (int q = 0; q < code.r; ++q) {
            if (coin(rng)) row |= 1ULL << (k + q);
        }
        code.row_masks[size_t(j)] = row;
    }
    code.syndrome_columns.assign(size_t(n), 0U);
    for (int col = 0; col < k; ++col) {
        uint32_t value = 0;
        for (int q = 0; q < code.r; ++q) {
            if ((code.row_masks[size_t(col)] >> (k + q)) & 1ULL) value ^= (1U << q);
        }
        code.syndrome_columns[size_t(col)] = value;
    }
    for (int q = 0; q < code.r; ++q) code.syndrome_columns[size_t(k + q)] = 1U << q;
    return code;
}

Code make_crc(int n, int k) {
    Code code;
    code.family = "named_crc_linear";
    code.n = n; code.k = k; code.r = n - k;
    auto [poly, pname] = crc_polynomial(code.r);
    code.name = "CRC_n" + std::to_string(n) + "_k" + std::to_string(k) + "_" + pname;
    code.row_masks.resize(size_t(k));
    for (int j = 0; j < k; ++j) {
        const uint64_t message = 1ULL << j;
        const uint64_t parity = crc_remainder(message, k, code.r, poly);
        code.row_masks[size_t(j)] = message | (parity << k);
    }
    code.syndrome_columns.assign(size_t(n), 0U);
    for (int col = 0; col < k; ++col) {
        uint32_t value = 0;
        for (int q = 0; q < code.r; ++q) {
            if ((code.row_masks[size_t(col)] >> (k + q)) & 1ULL) value ^= (1U << q);
        }
        code.syndrome_columns[size_t(col)] = value;
    }
    for (int q = 0; q < code.r; ++q) code.syndrome_columns[size_t(k + q)] = 1U << q;
    return code;
}

Code make_code(const std::string& family, int n, int k, std::mt19937_64& rng) {
    if (family == "RLC" || family == "random_linear") return make_random_linear(n, k, rng);
    if (family == "CRC" || family == "named_crc_linear") return make_crc(n, k);
    throw std::runtime_error("unknown family " + family);
}

uint64_t sample_message(int k, std::mt19937_64& rng) {
    return rng() & low_mask(k);
}

struct Sample {
    uint64_t transmitted = 0;
    uint64_t received = 0;
    int deleted_position = 0;
    uint64_t error_mask = 0;
};

Sample sample_channel(const Code& code, long double p, std::mt19937_64& rng) {
    Sample s;
    s.transmitted = code.encode(sample_message(code.k, rng));
    std::uniform_int_distribution<int> posdist(0, code.n - 1);
    s.deleted_position = posdist(rng);
    uint64_t survivor = delete_bit(s.transmitted, s.deleted_position, code.n);
    std::bernoulli_distribution flip{double(p)};
    for (int i = 0; i < code.n - 1; ++i) if (flip(rng)) s.error_mask |= 1ULL << i;
    s.received = survivor ^ s.error_mask;
    return s;
}

long double deletion_likelihood(uint64_t word, uint64_t received, int n, long double p, uint64_t* ops = nullptr) {
    const int m = n - 1;
    int d = 0;
    for (int i = 1; i < n; ++i) d += bit(word, i) != bit(received, i - 1);
    long double total = unnormalized_mass(d, m, p);
    uint64_t local_ops = uint64_t(n + m);
    for (int j = 0; j < n - 1; ++j) {
        d -= bit(word, j + 1) != bit(received, j);
        d += bit(word, j) != bit(received, j);
        total += unnormalized_mass(d, m, p);
        local_ops += 3;
    }
    if (ops) *ops += local_ops;
    return total / n;
}

struct Work {
    uint64_t histories = 0;
    uint64_t candidates = 0;
    uint64_t duplicates = 0;
    uint64_t membership = 0;
    uint64_t syndrome_bitops = 0;
    uint64_t exact_scores = 0;
    uint64_t likelihood_ops = 0;
    uint64_t heap_pushes = 0;
    uint64_t heap_pops = 0;
    uint64_t bound_checks = 0;
    uint64_t prefix_nodes = 0;
    uint64_t trellis_nodes = 0;
    uint64_t trellis_updates = 0;
    uint64_t trellis_terminals = 0;
    uint64_t peak_frontier = 0;
    uint64_t peak_seen = 0;
};

struct DecodeResult {
    bool certified = false;
    uint64_t decision = 0;
    std::vector<uint64_t> ties;
    long double best = -1;
    long double residual = 0;
    Work work;
    double wall_seconds = 0;
};

bool same_ties(std::vector<uint64_t> a, std::vector<uint64_t> b) {
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    return a == b;
}

struct StreamItem {
    long double probability = 0;
    uint64_t mask = 0;
    int hidden = 0;
    int weight = 0;
};

class ShellStream {
  public:
    ShellStream(int m_, long double p_, int streams_): m(m_), p(p_), streams(streams_) {
        reset_weight(0);
    }

    bool valid() const { return !exhausted; }

    StreamItem current() const {
        StreamItem item;
        item.probability = component_mass(weight, m, p, streams);
        item.mask = combination;
        item.hidden = hidden;
        item.weight = weight;
        return item;
    }

    void advance() {
        if (exhausted) return;
        if (hidden == 0) {
            hidden = 1;
            return;
        }
        hidden = 0;
        if (advance_combination()) return;
        reset_weight(weight + 1);
    }

  private:
    int m;
    long double p;
    int streams;
    int weight = 0;
    uint64_t combination = 0;
    int hidden = 0;
    bool exhausted = false;

    void reset_weight(int w) {
        weight = w;
        hidden = 0;
        if (w > m || (p == 0.0L && w > 0)) {
            exhausted = true;
            return;
        }
        exhausted = false;
        combination = (w == 0) ? 0ULL : ((1ULL << w) - 1ULL);
    }

    bool advance_combination() {
        if (weight == 0 || weight == m) return false;
        uint64_t x = combination;
        uint64_t u = x & (~x + 1ULL);
        uint64_t v = x + u;
        uint64_t next = v + (((v ^ x) / u) >> 2);
        const uint64_t limit = 1ULL << m;
        if (next >= limit) return false;
        combination = next;
        return true;
    }
};

struct HeapHistory {
    long double probability;
    int stream;
    uint64_t serial;
    StreamItem item;
};

struct HeapHistoryLess {
    bool operator()(const HeapHistory& a, const HeapHistory& b) const {
        if (a.probability != b.probability) return a.probability < b.probability;
        return a.serial > b.serial;
    }
};

DecodeResult fiber_decode(uint64_t received, const Code& code, long double p, uint64_t max_histories) {
    const auto start = Clock::now();
    const int n = code.n;
    const int m = n - 1;
    std::vector<ShellStream> streams;
    streams.reserve(size_t(n));
    for (int j = 0; j < n; ++j) streams.emplace_back(m, p, n);
    std::priority_queue<HeapHistory, std::vector<HeapHistory>, HeapHistoryLess> heap;
    std::vector<long double> eta(size_t(n), 0.0L);
    uint64_t serial = 0;
    Work work;
    for (int j = 0; j < n; ++j) {
        if (streams[size_t(j)].valid()) {
            auto item = streams[size_t(j)].current();
            eta[size_t(j)] = item.probability;
            heap.push({item.probability, j, serial++, item});
            work.heap_pushes++;
        }
    }
    long double residual = std::accumulate(eta.begin(), eta.end(), 0.0L);
    std::unordered_set<uint64_t> seen;
    seen.reserve(4096);
    std::vector<uint64_t> best_words;
    long double best = -1.0L;

    while (!heap.empty() && work.histories < max_histories) {
        auto top = heap.top(); heap.pop(); work.heap_pops++;
        const int j = top.stream;
        const long double old_eta = eta[size_t(j)];
        streams[size_t(j)].advance();
        if (streams[size_t(j)].valid()) {
            auto next = streams[size_t(j)].current();
            eta[size_t(j)] = next.probability;
            heap.push({next.probability, j, serial++, next});
            work.heap_pushes++;
        } else {
            eta[size_t(j)] = 0.0L;
        }
        residual += eta[size_t(j)] - old_eta;
        if (residual < 0 && residual > -1e-16L) residual = 0;
        work.histories++;
        uint64_t base = received ^ top.item.mask;
        uint64_t candidate = insert_bit(base, m, j, top.item.hidden);
        auto [it, inserted] = seen.insert(candidate);
        if (!inserted) {
            work.duplicates++;
        } else {
            work.candidates++;
            work.peak_seen = std::max<uint64_t>(work.peak_seen, seen.size());
            work.membership++;
            if (code.is_codeword(candidate, &work.syndrome_bitops)) {
                long double score = deletion_likelihood(candidate, received, n, p, &work.likelihood_ops);
                work.exact_scores++;
                if (score > best + TOL) {
                    best = score;
                    best_words = {candidate};
                } else if (std::fabs(score - best) <= TOL) {
                    best_words.push_back(candidate);
                }
            }
        }
        work.bound_checks++;
        work.peak_frontier = std::max<uint64_t>(work.peak_frontier, heap.size());
        if (!best_words.empty() && best > residual + TOL) {
            DecodeResult out;
            out.certified = true;
            out.decision = *std::min_element(best_words.begin(), best_words.end());
            std::sort(best_words.begin(), best_words.end());
            best_words.erase(std::unique(best_words.begin(), best_words.end()), best_words.end());
            out.ties = best_words;
            out.best = best;
            out.residual = residual;
            out.work = work;
            out.wall_seconds = std::chrono::duration<double>(Clock::now() - start).count();
            return out;
        }
    }
    DecodeResult out;
    out.certified = false;
    out.best = best;
    out.residual = residual;
    out.work = work;
    out.wall_seconds = std::chrono::duration<double>(Clock::now() - start).count();
    return out;
}

struct PrefixNode {
    long double bound = 0;
    int length = 0;
    uint64_t prefix = 0;
    std::array<uint8_t, 64> mismatches{};
    uint64_t expected_word = 0;
    bool expected_ready = false;
    uint64_t serial = 0;
};

struct PrefixLess {
    bool operator()(const PrefixNode& a, const PrefixNode& b) const {
        if (a.bound != b.bound) return a.bound < b.bound;
        if (a.length != b.length) return a.length < b.length;
        return a.serial > b.serial;
    }
};

long double prefix_bound(const std::array<uint8_t,64>& mismatches, int n, long double p) {
    const int m = n - 1;
    long double total = 0;
    for (int j = 0; j < n; ++j) total += unnormalized_mass(int(mismatches[size_t(j)]), m, p);
    return total / n;
}

DecodeResult prefix_decode(uint64_t received, const Code& code, long double p, uint64_t max_nodes) {
    const auto start = Clock::now();
    const int n = code.n;
    const int m = n - 1;
    PrefixNode root;
    root.bound = prefix_bound(root.mismatches, n, p);
    std::priority_queue<PrefixNode, std::vector<PrefixNode>, PrefixLess> heap;
    heap.push(root);
    uint64_t serial = 1;
    Work work;
    work.heap_pushes = 1;
    work.peak_frontier = 1;
    long double best = -1;
    std::vector<uint64_t> best_words;
    long double frontier = root.bound;

    while (!heap.empty() && work.prefix_nodes < max_nodes) {
        PrefixNode node = heap.top(); heap.pop();
        work.heap_pops++;
        work.prefix_nodes++;
        frontier = node.bound;
        if (!best_words.empty() && best > frontier + TOL) {
            DecodeResult out;
            out.certified = true;
            std::sort(best_words.begin(), best_words.end());
            best_words.erase(std::unique(best_words.begin(), best_words.end()), best_words.end());
            out.decision = best_words.front();
            out.ties = best_words;
            out.best = best;
            out.residual = frontier;
            out.work = work;
            out.wall_seconds = std::chrono::duration<double>(Clock::now() - start).count();
            return out;
        }
        if (node.length == n) {
            work.membership++;
            if (code.is_codeword(node.prefix, &work.syndrome_bitops)) {
                long double score = deletion_likelihood(node.prefix, received, n, p, &work.likelihood_ops);
                work.exact_scores++;
                if (score > best + TOL) {
                    best = score;
                    best_words = {node.prefix};
                } else if (std::fabs(score - best) <= TOL) {
                    best_words.push_back(node.prefix);
                }
            }
            continue;
        }
        const int pos = node.length;
        for (int b = 0; b < 2; ++b) {
            PrefixNode child = node;
            child.length = pos + 1;
            child.prefix = node.prefix | (uint64_t(b) << pos);
            child.serial = serial++;
            if (child.length == code.k) {
                child.expected_word = code.encode(child.prefix & low_mask(code.k));
                child.expected_ready = true;
            }
            if (child.length > code.k) {
                if (!child.expected_ready) {
                    child.expected_word = code.encode(child.prefix & low_mask(code.k));
                    child.expected_ready = true;
                }
                if (bit(child.expected_word, pos) != b) continue;
            }
            for (int j = 0; j < n; ++j) {
                if (pos == j) continue;
                const int out = pos < j ? pos : pos - 1;
                if (out >= 0 && out < m) child.mismatches[size_t(j)] += uint8_t(b != bit(received, out));
            }
            child.bound = prefix_bound(child.mismatches, n, p);
            work.bound_checks++;
            if (best_words.empty() || child.bound >= best - TOL) {
                heap.push(child);
                work.heap_pushes++;
            }
        }
        work.peak_frontier = std::max<uint64_t>(work.peak_frontier, heap.size());
    }
    if (!best_words.empty()) {
        frontier = heap.empty() ? 0.0L : heap.top().bound;
        if (best > frontier + TOL) {
            DecodeResult out;
            out.certified = true;
            std::sort(best_words.begin(), best_words.end());
            best_words.erase(std::unique(best_words.begin(), best_words.end()), best_words.end());
            out.decision = best_words.front();
            out.ties = best_words;
            out.best = best;
            out.residual = frontier;
            out.work = work;
            out.wall_seconds = std::chrono::duration<double>(Clock::now() - start).count();
            return out;
        }
    }
    DecodeResult out;
    out.certified = false;
    out.best = best;
    out.residual = frontier;
    out.work = work;
    out.wall_seconds = std::chrono::duration<double>(Clock::now() - start).count();
    return out;
}

DecodeResult exhaustive_decode(uint64_t received, const Code& code, long double p, uint64_t max_words = (1ULL<<20)) {
    if (code.k >= 63 || (1ULL << code.k) > max_words) throw std::runtime_error("exhaustive codebook too large");
    const auto start = Clock::now();
    DecodeResult out;
    out.best = -1;
    const uint64_t count = 1ULL << code.k;
    for (uint64_t msg = 0; msg < count; ++msg) {
        const uint64_t word = code.encode(msg);
        long double score = deletion_likelihood(word, received, code.n, p, &out.work.likelihood_ops);
        out.work.exact_scores++;
        if (score > out.best + TOL) {
            out.best = score;
            out.ties = {word};
        } else if (std::fabs(score - out.best) <= TOL) {
            out.ties.push_back(word);
        }
    }
    std::sort(out.ties.begin(), out.ties.end());
    out.ties.erase(std::unique(out.ties.begin(), out.ties.end()), out.ties.end());
    out.decision = out.ties.front();
    out.certified = true;
    out.wall_seconds = std::chrono::duration<double>(Clock::now() - start).count();
    return out;
}

struct TrellisNode {
    int f = 0;
    int g = 0;
    uint64_t serial = 0;
    int pos = 0;
    uint32_t syndrome = 0;
    uint64_t word = 0;
};

struct TrellisNodeGreater {
    bool operator()(const TrellisNode& a, const TrellisNode& b) const {
        if (a.f != b.f) return a.f > b.f;
        if (a.g != b.g) return a.g > b.g;
        return a.serial > b.serial;
    }
};

class AlignmentCodewordStream {
  public:
    AlignmentCodewordStream(uint64_t received_, int deleted_position_, const Code& code_, Work& work)
        : received(received_), deleted_position(deleted_position_), code(code_), n(code.n), r(code.r), states(1U << code.r) {
        target.assign(size_t(n), -1);
        int yi = 0;
        for (int pos = 0; pos < n; ++pos) {
            if (pos == deleted_position) target[size_t(pos)] = -1;
            else target[size_t(pos)] = bit(received, yi++);
        }
        const uint16_t inf = uint16_t(n + 1);
        dp.assign(size_t(n + 1) * states, inf);
        at(n, 0) = 0;
        for (int pos = n - 1; pos >= 0; --pos) {
            const uint32_t column = code.syndrome_columns[size_t(pos)];
            for (uint32_t required = 0; required < states; ++required) {
                const int c0 = (target[size_t(pos)] < 0 || target[size_t(pos)] == 0) ? 0 : 1;
                const int c1 = (target[size_t(pos)] < 0 || target[size_t(pos)] == 1) ? 0 : 1;
                const int v0 = c0 + int(at(pos + 1, required));
                const int v1 = c1 + int(at(pos + 1, required ^ column));
                at(pos, required) = uint16_t(std::min(v0, v1));
                work.trellis_updates += 2;
            }
        }
        TrellisNode root;
        root.f = int(at(0, 0));
        root.g = 0;
        root.serial = serial++;
        root.pos = 0;
        root.syndrome = 0;
        root.word = 0;
        heap.push(root);
        work.heap_pushes++;
    }

    std::optional<std::pair<uint64_t,int>> next(Work& work) {
        while (!heap.empty()) {
            TrellisNode node = heap.top(); heap.pop();
            work.heap_pops++;
            work.trellis_nodes++;
            if (node.pos == n) {
                if (node.syndrome == 0) {
                    work.trellis_terminals++;
                    return std::make_pair(node.word, node.g);
                }
                continue;
            }
            const int pos = node.pos;
            const uint32_t column = code.syndrome_columns[size_t(pos)];
            for (int b = 0; b < 2; ++b) {
                const uint32_t next_syndrome = node.syndrome ^ (b ? column : 0U);
                const int cost = (target[size_t(pos)] < 0 || target[size_t(pos)] == b) ? 0 : 1;
                const int ng = node.g + cost;
                const int nh = int(at(pos + 1, next_syndrome));
                if (nh > n) continue;
                TrellisNode child;
                child.f = ng + nh;
                child.g = ng;
                child.serial = serial++;
                child.pos = pos + 1;
                child.syndrome = next_syndrome;
                child.word = node.word | (uint64_t(b) << pos);
                heap.push(child);
                work.heap_pushes++;
            }
            work.peak_frontier = std::max<uint64_t>(work.peak_frontier, heap.size());
        }
        return std::nullopt;
    }

  private:
    uint64_t received;
    int deleted_position;
    const Code& code;
    int n;
    int r;
    uint32_t states;
    std::vector<int8_t> target;
    std::vector<uint16_t> dp;
    std::priority_queue<TrellisNode, std::vector<TrellisNode>, TrellisNodeGreater> heap;
    uint64_t serial = 0;

    uint16_t& at(int pos, uint32_t state) { return dp[size_t(pos) * states + state]; }
    const uint16_t& at(int pos, uint32_t state) const { return dp[size_t(pos) * states + state]; }
};

struct TrellisHead {
    long double probability = 0;
    int stream = 0;
    uint64_t serial = 0;
    uint64_t word = 0;
    int cost = 0;
};

struct TrellisHeadLess {
    bool operator()(const TrellisHead& a, const TrellisHead& b) const {
        if (a.probability != b.probability) return a.probability < b.probability;
        return a.serial > b.serial;
    }
};

bool syndrome_trellis_feasible(const Code& code, uint64_t max_dp_updates) {
    if (code.r >= 31) return false;
    const long double estimate = 2.0L * code.n * code.n * (uint64_t(1) << code.r);
    return estimate <= static_cast<long double>(max_dp_updates);
}

DecodeResult syndrome_trellis_decode(
    uint64_t received,
    const Code& code,
    long double p,
    uint64_t max_terminals,
    uint64_t max_dp_updates
) {
    DecodeResult out;
    if (!syndrome_trellis_feasible(code, max_dp_updates)) return out;
    const auto start = Clock::now();
    Work work;
    std::vector<std::unique_ptr<AlignmentCodewordStream>> streams;
    streams.reserve(size_t(code.n));
    for (int j = 0; j < code.n; ++j) {
        streams.push_back(std::make_unique<AlignmentCodewordStream>(received, j, code, work));
    }
    std::priority_queue<TrellisHead, std::vector<TrellisHead>, TrellisHeadLess> heap;
    std::vector<long double> eta(size_t(code.n), 0.0L);
    uint64_t serial = 0;
    for (int j = 0; j < code.n; ++j) {
        auto item = streams[size_t(j)]->next(work);
        if (item) {
            const auto [word,cost] = *item;
            const long double probability = component_mass(cost, code.n - 1, p, code.n);
            eta[size_t(j)] = probability;
            heap.push({probability,j,serial++,word,cost});
            work.heap_pushes++;
        }
    }
    long double residual = std::accumulate(eta.begin(), eta.end(), 0.0L);
    std::unordered_set<uint64_t> seen;
    seen.reserve(4096);
    long double best = -1.0L;
    std::vector<uint64_t> best_words;
    while (!heap.empty() && work.trellis_terminals <= max_terminals) {
        auto top = heap.top(); heap.pop(); work.heap_pops++;
        const int j = top.stream;
        const long double old_eta = eta[size_t(j)];
        auto next = streams[size_t(j)]->next(work);
        if (next) {
            const auto [next_word,next_cost] = *next;
            const long double probability = component_mass(next_cost, code.n - 1, p, code.n);
            eta[size_t(j)] = probability;
            heap.push({probability,j,serial++,next_word,next_cost});
            work.heap_pushes++;
        } else eta[size_t(j)] = 0.0L;
        residual = std::max(0.0L, residual + eta[size_t(j)] - old_eta);
        if (seen.insert(top.word).second) {
            work.candidates++;
            uint64_t ops = 0;
            const long double score = deletion_likelihood(top.word, received, code.n, p, &ops);
            work.exact_scores++;
            work.likelihood_ops += ops;
            if (score > best + TOL) {
                best = score;
                best_words = {top.word};
            } else if (std::fabs(score - best) <= TOL) best_words.push_back(top.word);
        } else work.duplicates++;
        work.bound_checks++;
        if (!best_words.empty() && best > residual + TOL) {
            out.certified = true;
            break;
        }
    }
    std::sort(best_words.begin(), best_words.end());
    best_words.erase(std::unique(best_words.begin(), best_words.end()), best_words.end());
    if (!best_words.empty()) out.decision = best_words.front();
    out.ties = best_words;
    out.best = best;
    out.residual = residual;
    out.work = work;
    out.wall_seconds = std::chrono::duration<double>(Clock::now() - start).count();
    return out;
}

struct Args {
    std::string output = "compiled_trials.csv";
    uint64_t seed = 20260806ULL;
    std::vector<int> ns{32,48,64};
    std::vector<double> rates{0.75,0.875};
    std::vector<double> ps{0.002,0.005,0.01,0.02};
    std::vector<std::string> families{"RLC","CRC"};
    int trials32 = 80;
    int trials48 = 50;
    int trials64 = 30;
    int timing_repeats = 3;
    uint64_t max_histories = 5000000;
    uint64_t max_prefix_nodes = 5000000;
    uint64_t max_trellis_terminals = 1000000;
    uint64_t max_trellis_dp_updates = 50000000;
    bool self_test = false;
};

std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) if (!item.empty()) out.push_back(item);
    return out;
}

template <class T>
std::vector<T> parse_numeric_list(const std::string& s) {
    std::vector<T> out;
    for (const auto& item: split(s, ',')) {
        std::stringstream ss(item);
        T value{}; ss >> value;
        if (!ss) throw std::runtime_error("bad numeric list: " + s);
        out.push_back(value);
    }
    return out;
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        auto need = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + key);
            return argv[++i];
        };
        if (key == "--output") a.output = need();
        else if (key == "--seed") a.seed = std::stoull(need());
        else if (key == "--n") a.ns = parse_numeric_list<int>(need());
        else if (key == "--rates") a.rates = parse_numeric_list<double>(need());
        else if (key == "--p") a.ps = parse_numeric_list<double>(need());
        else if (key == "--families") a.families = split(need(), ',');
        else if (key == "--trials32") a.trials32 = std::stoi(need());
        else if (key == "--trials48") a.trials48 = std::stoi(need());
        else if (key == "--trials64") a.trials64 = std::stoi(need());
        else if (key == "--timing-repeats") a.timing_repeats = std::stoi(need());
        else if (key == "--max-histories") a.max_histories = std::stoull(need());
        else if (key == "--max-prefix-nodes") a.max_prefix_nodes = std::stoull(need());
        else if (key == "--max-trellis-terminals") a.max_trellis_terminals = std::stoull(need());
        else if (key == "--max-trellis-dp-updates") a.max_trellis_dp_updates = std::stoull(need());
        else if (key == "--self-test") a.self_test = true;
        else throw std::runtime_error("unknown argument " + key);
    }
    return a;
}

int trials_for_n(const Args& a, int n) {
    if (n <= 32) return a.trials32;
    if (n <= 48) return a.trials48;
    return a.trials64;
}

template <class F>
std::pair<DecodeResult,double> repeated(F&& fn, int repeats) {
    std::vector<double> times;
    DecodeResult first;
    for (int i = 0; i < repeats; ++i) {
        DecodeResult r = fn();
        if (i == 0) first = r;
        else if (r.certified != first.certified || (r.certified && !same_ties(r.ties, first.ties))) {
            throw std::runtime_error("non-deterministic decoder output across timing repeats");
        }
        times.push_back(r.wall_seconds);
    }
    std::sort(times.begin(), times.end());
    return {first, times[times.size()/2]};
}

bool self_test() {
    std::mt19937_64 rng(12345);
    for (const std::string family: {"RLC", "CRC"}) {
        for (int n: {8,10,12}) {
            int k = n - 3;
            Code code = make_code(family, n, k, rng);
            for (double pd: {0.0, 0.05, 0.2}) {
                for (int trial = 0; trial < 50; ++trial) {
                    Sample s = sample_channel(code, pd, rng);
                    DecodeResult f = fiber_decode(s.received, code, pd, 2000000);
                    DecodeResult p = prefix_decode(s.received, code, pd, 2000000);
                    DecodeResult tr = syndrome_trellis_decode(s.received, code, pd, 2000000, 50000000);
                    DecodeResult e = exhaustive_decode(s.received, code, pd);
                    if (!f.certified || !p.certified || !tr.certified || !same_ties(f.ties,e.ties) || !same_ties(p.ties,e.ties) || !same_ties(tr.ties,e.ties)) {
                        std::cerr << "SELF_TEST_FAIL family=" << family << " n=" << n << " p=" << pd << " trial=" << trial << "\n";
                        return false;
                    }
                }
            }
        }
    }
    std::cout << "SELF_TEST_PASS cases=600\n";
    return true;
}

void write_header(std::ofstream& out) {
    out << "family,code_name,rate,n,k,p,trial,error_weight,deleted_position,agreement,fiber_certified,prefix_certified,trellis_available,trellis_certified,"
        << "fiber_wall,prefix_wall,trellis_wall,best_baseline_wall,best_baseline_name,fiber_over_prefix_wall,fiber_over_best_wall,"
        << "fiber_histories,fiber_candidates,fiber_duplicates,fiber_membership,fiber_exact_scores,fiber_peak_seen,fiber_peak_frontier,"
        << "prefix_nodes,prefix_membership,prefix_exact_scores,prefix_peak_frontier,trellis_nodes,trellis_updates,trellis_terminals,trellis_exact_scores\n";
}

int run_benchmark(const Args& a) {
    std::mt19937_64 rng(a.seed);
    std::ofstream out(a.output);
    if (!out) throw std::runtime_error("cannot open output " + a.output);
    out << std::setprecision(17);
    write_header(out);
    uint64_t rows = 0;
    uint64_t disagreements = 0;
    uint64_t failures = 0;
    for (const auto& family: a.families) {
        for (double rate: a.rates) {
            for (int n: a.ns) {
                if (n > 64 || n < 8) throw std::runtime_error("compiled benchmark supports 8 <= n <= 64");
                int k = std::max(1, std::min(n-1, int(std::llround(rate * n))));
                Code code = make_code(family, n, k, rng);
                for (double p: a.ps) {
                    const int trials = trials_for_n(a, n);
                    std::cout << "compiled family=" << code.family << " R=" << double(k)/n << " n=" << n << " p=" << p << " trials=" << trials << std::endl;
                    for (int trial = 0; trial < trials; ++trial) {
                        Sample s = sample_channel(code, p, rng);
                        bool fiber_first = (rng() & 1ULL) != 0;
                        DecodeResult f, pr, tr;
                        double ft=0, pt=0, tt=std::numeric_limits<double>::infinity();
                        if (fiber_first) {
                            auto fm = repeated([&]{ return fiber_decode(s.received, code, p, a.max_histories); }, a.timing_repeats);
                            f=fm.first; ft=fm.second;
                            auto pm = repeated([&]{ return prefix_decode(s.received, code, p, a.max_prefix_nodes); }, a.timing_repeats);
                            pr=pm.first; pt=pm.second;
                        } else {
                            auto pm = repeated([&]{ return prefix_decode(s.received, code, p, a.max_prefix_nodes); }, a.timing_repeats);
                            pr=pm.first; pt=pm.second;
                            auto fm = repeated([&]{ return fiber_decode(s.received, code, p, a.max_histories); }, a.timing_repeats);
                            f=fm.first; ft=fm.second;
                        }
                        const bool trellis_available = syndrome_trellis_feasible(code, a.max_trellis_dp_updates);
                        if (trellis_available) {
                            auto tm = repeated([&]{ return syndrome_trellis_decode(s.received, code, p, a.max_trellis_terminals, a.max_trellis_dp_updates); }, a.timing_repeats);
                            tr=tm.first; tt=tm.second;
                        }
                        const bool any_baseline = pr.certified || tr.certified;
                        bool agreement = f.certified && any_baseline
                            && (!pr.certified || same_ties(f.ties, pr.ties))
                            && (!tr.certified || same_ties(f.ties, tr.ties));
                        if (!agreement) disagreements++;
                        if (!f.certified || !any_baseline) failures++;
                        const double effective_prefix_wall = pr.certified ? pt : std::numeric_limits<double>::infinity();
                        const double effective_trellis_wall = tr.certified ? tt : std::numeric_limits<double>::infinity();
                        const double best_wall = std::min(effective_prefix_wall, effective_trellis_wall);
                        const char* best_name = (effective_trellis_wall < effective_prefix_wall) ? "SYNDROME_TRELLIS" : "PREFIX_ASTAR";
                        out << code.family << ',' << '"' << code.name << '"' << ',' << double(k)/n << ',' << n << ',' << k << ',' << p << ',' << trial << ','
                            << __builtin_popcountll(s.error_mask) << ',' << s.deleted_position << ',' << (agreement?1:0) << ',' << (f.certified?1:0) << ',' << (pr.certified?1:0) << ','
                            << (trellis_available?1:0) << ',' << (trellis_available && tr.certified?1:0) << ','
                            << ft << ',' << pt << ',' << (trellis_available?tt:0.0) << ',' << best_wall << ',' << best_name << ','
                            << (pt>0?ft/pt:std::numeric_limits<double>::infinity()) << ',' << (best_wall>0?ft/best_wall:std::numeric_limits<double>::infinity()) << ','
                            << f.work.histories << ',' << f.work.candidates << ',' << f.work.duplicates << ',' << f.work.membership << ',' << f.work.exact_scores << ','
                            << f.work.peak_seen << ',' << f.work.peak_frontier << ',' << pr.work.prefix_nodes << ',' << pr.work.membership << ',' << pr.work.exact_scores << ',' << pr.work.peak_frontier << ','
                            << tr.work.trellis_nodes << ',' << tr.work.trellis_updates << ',' << tr.work.trellis_terminals << ',' << tr.work.exact_scores << '\n';
                        rows++;
                    }
                }
            }
        }
    }
    out.close();
    std::cout << "COMPILED_BENCHMARK_DONE rows=" << rows << " disagreements=" << disagreements << " failures=" << failures << "\n";
    return (disagreements==0 && failures==0) ? 0 : 3;
}

} // namespace fg

int main(int argc, char** argv) {
    try {
        fg::Args args = fg::parse_args(argc, argv);
        if (args.self_test) return fg::self_test() ? 0 : 2;
        return fg::run_benchmark(args);
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
