#define main original_flagship_benchmark_main
#include "../../FIBER_GRAND_PRE_MANUSCRIPT_CLOSURE_v5_0/cpp/flagship_benchmark.cpp"
#undef main

#include <set>
#include <tuple>

namespace {

struct ReplayTarget {
    std::string family_arg;
    std::string family_output;
    double p;
    int trial;
};

const std::vector<ReplayTarget> TARGETS = {
    {"RLC", "random_linear", 0.005, 561},
    {"RLC", "random_linear", 0.010, 31},
    {"CRC", "named_crc_linear", 0.005, 382},
    {"CRC", "named_crc_linear", 0.010, 105},
    {"CRC", "named_crc_linear", 0.010, 185},
    {"CRC", "named_crc_linear", 0.010, 296},
    {"CRC", "named_crc_linear", 0.010, 314},
};

bool is_target(const std::string& family, double p, int trial) {
    for (const auto& t : TARGETS) {
        if (t.family_arg == family && std::fabs(t.p - p) < 1e-15 && t.trial == trial) return true;
    }
    return false;
}

std::string hex_word(uint64_t value) {
    std::ostringstream ss;
    ss << "0x" << std::hex << value;
    return ss.str();
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::cerr << "usage: replay_failures OUTPUT.csv [timing_repeats]\n";
            return 1;
        }
        const std::string output_path = argv[1];
        const int repeats = argc >= 3 ? std::stoi(argv[2]) : 3;
        constexpr int n = 64;
        constexpr int k = 48;
        constexpr int warmup_trials = 3;
        constexpr int ordinary_trials = 600;
        constexpr uint64_t shell3_histories = 5341184ULL;
        constexpr uint64_t prefix_cap = 10000000ULL;
        constexpr uint64_t seed = 20560816ULL;

        std::ofstream out(output_path);
        if (!out) throw std::runtime_error("cannot open replay output");
        out << std::setprecision(17);
        out << "family,code_name,rate,n,k,p,trial,error_weight,deleted_position,transmitted_hex,received_hex,"
               "agreement,fiber_certified,prefix_certified,fiber_wall,prefix_wall,fiber_over_prefix_wall,"
               "fiber_histories,fiber_candidates,fiber_duplicates,fiber_membership,fiber_exact_scores,fiber_peak_seen,fiber_peak_frontier,"
               "prefix_nodes,prefix_membership,prefix_exact_scores,prefix_peak_frontier,shell3_history_cap,prefix_node_cap\n";

        std::mt19937_64 rng(seed);
        int replayed = 0;
        for (const std::string& family : {std::string("RLC"), std::string("CRC")}) {
            fg::Code code = fg::make_code(family, n, k, rng);
            for (double p : {0.002, 0.005, 0.010}) {
                for (int warm = 0; warm < warmup_trials; ++warm) {
                    (void)fg::sample_channel(code, p, rng);
                    // Decoder warmups consume no random numbers in the original campaign.
                }
                for (int trial = 0; trial < ordinary_trials; ++trial) {
                    fg::Sample sample = fg::sample_channel(code, p, rng);
                    std::vector<int> decoder_order{0, 1};
                    std::shuffle(decoder_order.begin(), decoder_order.end(), rng);
                    if (!is_target(family, p, trial)) continue;

                    std::cout << "TARGET_START family=" << code.family << " p=" << p << " trial=" << trial
                              << " E=" << __builtin_popcountll(sample.error_mask)
                              << " deleted=" << sample.deleted_position << std::endl;
                    fg::DecodeResult fiber;
                    fg::DecodeResult prefix;
                    double fiber_time = 0.0;
                    double prefix_time = 0.0;
                    for (int which : decoder_order) {
                        if (which == 0) {
                            auto timed = fg::repeated(
                                [&] { return fg::fiber_decode(sample.received, code, p, shell3_histories); },
                                repeats
                            );
                            fiber = timed.first;
                            fiber_time = timed.second;
                        } else {
                            auto timed = fg::repeated(
                                [&] { return fg::prefix_decode(sample.received, code, p, prefix_cap); },
                                repeats
                            );
                            prefix = timed.first;
                            prefix_time = timed.second;
                        }
                    }
                    const bool agreement = fiber.certified && prefix.certified && fg::same_ties(fiber.ties, prefix.ties);
                    out << code.family << ',' << '"' << code.name << '"' << ',' << double(k) / n << ',' << n << ',' << k << ',' << p << ',' << trial << ','
                        << __builtin_popcountll(sample.error_mask) << ',' << sample.deleted_position << ','
                        << '"' << hex_word(sample.transmitted) << '"' << ',' << '"' << hex_word(sample.received) << '"' << ','
                        << (agreement ? 1 : 0) << ',' << (fiber.certified ? 1 : 0) << ',' << (prefix.certified ? 1 : 0) << ','
                        << fiber_time << ',' << prefix_time << ','
                        << (prefix_time > 0.0 ? fiber_time / prefix_time : std::numeric_limits<double>::infinity()) << ','
                        << fiber.work.histories << ',' << fiber.work.candidates << ',' << fiber.work.duplicates << ',' << fiber.work.membership << ','
                        << fiber.work.exact_scores << ',' << fiber.work.peak_seen << ',' << fiber.work.peak_frontier << ','
                        << prefix.work.prefix_nodes << ',' << prefix.work.membership << ',' << prefix.work.exact_scores << ',' << prefix.work.peak_frontier << ','
                        << shell3_histories << ',' << prefix_cap << '\n';
                    out.flush();
                    std::cout << "REPLAY family=" << code.family << " p=" << p << " trial=" << trial
                              << " E=" << __builtin_popcountll(sample.error_mask)
                              << " fiber_certified=" << (fiber.certified ? 1 : 0)
                              << " prefix_certified=" << (prefix.certified ? 1 : 0)
                              << " agreement=" << (agreement ? 1 : 0)
                              << " fiber_histories=" << fiber.work.histories
                              << " prefix_nodes=" << prefix.work.prefix_nodes << std::endl;
                    ++replayed;
                }
            }
        }
        out.close();
        if (replayed != int(TARGETS.size())) {
            std::cerr << "expected " << TARGETS.size() << " targets but replayed " << replayed << "\n";
            return 2;
        }
        std::cout << "REPLAY_DONE targets=" << replayed << "\n";
        return 0; // Scientific pass/fail is adjudicated from the CSV, never from process exit here.
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
