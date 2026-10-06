#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Vec = std::array<double, 3>;
using Counts = std::array<int, 3>;

struct Options {
    std::string output;
    double beta = 0.5;
    int sample_size = 1;  // 0 denotes sufficient feedback.
    int population = 1000;
    int seeds = 10;
    double horizon = 20.0;
    double record_dt = 1.0;
    std::uint64_t base_seed = 1;
    Vec plus{};
    Vec minus{};
};

Vec parse_vec(const std::string& text) {
    Vec result{};
    std::stringstream stream(text);
    std::string token;
    for (int i = 0; i < 3; ++i) {
        if (!std::getline(stream, token, ',')) throw std::runtime_error("invalid vector");
        result[i] = std::stod(token);
    }
    return result;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) throw std::runtime_error("missing option value");
        std::string key = argv[i], value = argv[i + 1];
        if (key == "--output") options.output = value;
        else if (key == "--beta") options.beta = std::stod(value);
        else if (key == "--m") options.sample_size = value == "inf" ? 0 : std::stoi(value);
        else if (key == "--population") options.population = std::stoi(value);
        else if (key == "--seeds") options.seeds = std::stoi(value);
        else if (key == "--horizon") options.horizon = std::stod(value);
        else if (key == "--record-dt") options.record_dt = std::stod(value);
        else if (key == "--base-seed") options.base_seed = std::stoull(value);
        else if (key == "--x-plus") options.plus = parse_vec(value);
        else if (key == "--x-minus") options.minus = parse_vec(value);
        else throw std::runtime_error("unknown option: " + key);
    }
    if (options.output.empty() || options.population < 3 || options.seeds < 1 ||
        options.beta <= 0.0 || options.horizon <= 0.0 || options.record_dt <= 0.0 ||
        (options.sample_size != 0 && options.sample_size != 1)) {
        throw std::runtime_error("invalid options");
    }
    return options;
}

Counts initial_counts(const Vec& state, int population) {
    Counts counts{};
    std::array<double, 3> residual{};
    int assigned = 0;
    for (int i = 0; i < 3; ++i) {
        double raw = state[i] * population;
        counts[i] = static_cast<int>(std::floor(raw));
        residual[i] = raw - counts[i];
        assigned += counts[i];
    }
    while (assigned < population) {
        int best = static_cast<int>(std::max_element(residual.begin(), residual.end()) - residual.begin());
        ++counts[best];
        residual[best] = -1.0;
        ++assigned;
    }
    return counts;
}

template <class RNG>
int draw_strategy(const Counts& counts, int pool_size, RNG& rng) {
    std::uniform_int_distribution<int> uniform(0, pool_size - 1);
    int draw = uniform(rng), cumulative = 0;
    for (int strategy = 0; strategy < 3; ++strategy) {
        cumulative += counts[strategy];
        if (draw < cumulative) return strategy;
    }
    return 2;
}

double logistic(double value) {
    if (value >= 0.0) return 1.0 / (1.0 + std::exp(-value));
    double exponential = std::exp(value);
    return exponential / (1.0 + exponential);
}

void simulate(
    const Options& options,
    const Vec& initial,
    const std::string& direction,
    int seed_index,
    std::ofstream& output
) {
    constexpr double A[3][3] = {
        {0.0, -0.37061410, 2.77335386},
        {0.24483140, 0.0, -2.70698466},
        {-2.08439346, 3.52730968, 0.0},
    };
    // Common random numbers couple the plus/minus perturbations and sharply
    // reduce the variance of their signed response.  Each marginal process
    // remains an exact realization of the same finite-population chain.
    std::uint64_t seed = options.base_seed + 1000003ULL * seed_index + 17ULL;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uniform01(0.0, 1.0);
    Counts counts = initial_counts(initial, options.population);
    int rows = static_cast<int>(std::llround(options.horizon / options.record_dt)) + 1;
    double event_scale = 2.0 / options.beta;
    std::int64_t total_events = static_cast<std::int64_t>(
        std::llround(options.horizon * options.population * event_scale)
    );
    int next_row = 0;
    auto record = [&](double time) {
        int minimum = *std::min_element(counts.begin(), counts.end());
        output << direction << ',' << seed_index << ',' << std::setprecision(17) << time;
        for (int value : counts) output << ',' << static_cast<double>(value) / options.population;
        output << ',' << minimum << '\n';
    };
    record(0.0);
    next_row = 1;
    for (std::int64_t event = 1; event <= total_events; ++event) {
        int old_strategy = draw_strategy(counts, options.population, rng);
        Counts model_pool = counts;
        --model_pool[old_strategy];
        int candidate = draw_strategy(model_pool, options.population - 1, rng);
        double payoff_old = 0.0, payoff_candidate = 0.0;
        if (options.sample_size == 1) {
            Counts old_pool = counts;
            Counts candidate_pool = counts;
            --old_pool[old_strategy];
            --candidate_pool[candidate];
            int opponent_old = draw_strategy(old_pool, options.population - 1, rng);
            int opponent_candidate = draw_strategy(candidate_pool, options.population - 1, rng);
            payoff_old = A[old_strategy][opponent_old];
            payoff_candidate = A[candidate][opponent_candidate];
        } else {
            for (int opponent = 0; opponent < 3; ++opponent) {
                int old_count = counts[opponent] - (opponent == old_strategy ? 1 : 0);
                int candidate_count = counts[opponent] - (opponent == candidate ? 1 : 0);
                payoff_old += A[old_strategy][opponent] * old_count / (options.population - 1.0);
                payoff_candidate += A[candidate][opponent] * candidate_count / (options.population - 1.0);
            }
        }
        // Consume one revision draw on every event. Together with fixed
        // category draws, this keeps plus/minus streams synchronized.
        double revision_draw = uniform01(rng);
        if (old_strategy != candidate &&
            revision_draw < logistic(options.beta * (payoff_candidate - payoff_old))) {
            --counts[old_strategy];
            ++counts[candidate];
        }
        while (next_row < rows) {
            std::int64_t target = static_cast<std::int64_t>(std::llround(
                next_row * options.record_dt * options.population * event_scale
            ));
            if (event < target) break;
            record(next_row * options.record_dt);
            ++next_row;
        }
    }
    while (next_row < rows) {
        record(next_row * options.record_dt);
        ++next_row;
    }
}

int main(int argc, char** argv) {
    try {
        Options options = parse_options(argc, argv);
        std::ofstream output(options.output);
        if (!output) throw std::runtime_error("cannot open output");
        output << "direction,seed,time,x1,x2,x3,min_count\n";
        for (int seed = 0; seed < options.seeds; ++seed) {
            simulate(options, options.plus, "plus", seed, output);
            simulate(options, options.minus, "minus", seed, output);
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
