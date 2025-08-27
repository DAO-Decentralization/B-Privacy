#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdint>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <random>
#include <unordered_set>
#include <map>
#include <cstring>

using namespace std;

// Global variables
bool g_addNoise = false;
mt19937 g_rng;

// Function to generate Laplace noise with scale b
double generateLaplaceNoise(double b, mt19937& gen) {
    uniform_real_distribution<double> uniform(-0.5, 0.5);
    double u = uniform(gen);
    
    // Use inverse CDF of Laplace distribution
    if (u < 0) {
        return b * log(1 + 2 * u);
    } else {
        return -b * log(1 - 2 * u);
    }
}


// Structure to hold voter data
struct Voter {
    string voter_address;
    string choice;
    double voting_power;
    int mapped_choice; // 1, 2, 3, ... for each option, 0 for others
};

// Structure to hold proposal data
struct Proposal {
    string proposal_id;
    string dao_id;
    vector<Voter> voters;
    double total_votes;
    vector<string> option_texts; // Text of each option
    vector<int> option_counts;   // Count of voters for each option
    vector<float> option_votes;  // Voting power for each option
};

// Structure to hold attack results
struct AttackResult {
    string proposal_id;
    string dao_id;
    int total_voters;
    int valid_voters;
    double total_voting_power;
    int whale_deanonymized;
    int whale_correct;
    double whale_voting_power;
    double whale_voting_power_correct;
    int precision_deanonymized;
    int precision_correct;
    double precision_voting_power;
    double precision_voting_power_correct;
    int total_deanonymized;
    int total_correct;
    double total_voting_power_deanonymized;
    double total_voting_power_correct;
    double accuracy;
    vector<string> option_texts;
    vector<int> option_counts;
    vector<float> option_votes;
};

// Function to map choices to all options in a proposal
void mapChoicesForProposal(Proposal& proposal) {
    // Count frequency of each choice
    map<string, pair<int, float>> choiceCount;
    for (const auto& voter : proposal.voters) {
        choiceCount[voter.choice].first++;
        choiceCount[voter.choice].second += voter.voting_power;
    }
    
    // Sort choices by frequency (descending)
    vector<pair<string, pair<int, float>>> sortedChoices;
    for (const auto& [choice, count] : choiceCount) {
        sortedChoices.push_back({choice, count});
    }
    sort(sortedChoices.begin(), sortedChoices.end(), 
         [](const pair<string, pair<int, float>>& a, const pair<string, pair<int, float>>& b) {
             return a.second.second > b.second.second;
         });
    
    // Store all options
    proposal.option_texts.clear();
    proposal.option_counts.clear();
    proposal.option_votes.clear();
    
    for (const auto& [choice, count] : sortedChoices) {
        proposal.option_texts.push_back(choice);
        proposal.option_counts.push_back(count.first);
        proposal.option_votes.push_back(count.second);
    }
    
    // Map each voter's choice to option number (1-indexed)
    for (auto& voter : proposal.voters) {
        voter.mapped_choice = 0; // Default to 0 (not found)
        for (size_t i = 0; i < proposal.option_texts.size(); i++) {
            if (voter.choice == proposal.option_texts[i]) {
                voter.mapped_choice = i + 1; // 1-indexed
                break;
            }
        }
    }
}

// Function to run whale attack on a proposal
tuple<int, int, double, double> runWhaleAttack(const Proposal& proposal, vector<int>& whale_guesses, vector<bool>& is_whale_guess) {
    int num_options = proposal.option_texts.size();
    if (num_options < 2) return {proposal.voters.size(), proposal.voters.size(), proposal.total_votes, proposal.total_votes};
    
    // Get valid voters and their data
    vector<int> voter_indices;
    vector<double> voter_weights;
    vector<int> actual_votes;
    
    for (size_t i = 0; i < proposal.voters.size(); i++) {
        if (proposal.voters[i].mapped_choice > 0) {
            voter_indices.push_back(i);
            voter_weights.push_back(proposal.voters[i].voting_power);
            actual_votes.push_back(proposal.voters[i].mapped_choice);
        }
    }
    
    if (voter_indices.empty()) return {0, 0, 0.0, 0.0};
    
    // Initialize current tallies for each option
    vector<double> current_tallies(num_options, 0.0);
    for (size_t i = 0; i < proposal.option_counts.size(); i++) {
        // Calculate tally from voting power, not just count
        for (const auto& voter : proposal.voters) {
            if (voter.mapped_choice == (int)(i + 1)) {
                current_tallies[i] += voter.voting_power;
            }
        }
    }
    
    // Add noise to tallies if flag is set
    if (g_addNoise && current_tallies.size() >= 2) {
        // Calculate total voting power
        double total_power = 0.0;
        for (double tally : current_tallies) {
            total_power += tally;
        }
        
        // Calculate b = 0.1 * total_voting_power / ln(20)
        double b = 0.1 * total_power / log(20.0);
        
        // Generate noise for winning option (option 0, highest voting power)
        double noise = generateLaplaceNoise(b, g_rng);
        
        // Add noise to winning option
        current_tallies[0] += noise;
        
        // Subtract noise from second place option (reverse direction)
        current_tallies[1] -= noise;
        
        // Ensure second place doesn't go negative
        if (current_tallies[1] < 0) {
            current_tallies[1] = 0;
        }
    }
    
    // Initialize tracking arrays
    whale_guesses.assign(proposal.voters.size(), 0);
    is_whale_guess.assign(proposal.voters.size(), false);
    
    // Calculate noise buffer once if needed
    double noise_buffer = 0.0;
    if (g_addNoise) {
        double total_power = 0.0;
        for (double tally : current_tallies) {
            total_power += tally;
        }
        noise_buffer = 0.1 * total_power;
    }
    
    unordered_set<int> identified_voters;
    int total_whales = 0;
    int correct_whales = 0;
    double total_whale_power = 0.0;
    double correct_whale_power = 0.0;
    bool made_progress = true;
    
    while (made_progress) {
        made_progress = false;
        
        for (size_t idx = 0; idx < voter_indices.size(); idx++) {
            int voter_idx = voter_indices[idx];
            if (identified_voters.count(voter_idx)) continue;
            
            double voter_weight = voter_weights[idx];
            vector<bool> possible_votes(num_options, true);
            
            // Check which options this voter could NOT have voted for
            for (int option = 0; option < num_options; option++) {
                double threshold = current_tallies[option] + noise_buffer;
                
                if (voter_weight > threshold) {
                    possible_votes[option] = false;
                }
            }
            
            // Count how many options are still possible
            int possible_count = 0;
            int possible_option = -1;
            for (int option = 0; option < num_options; option++) {
                if (possible_votes[option]) {
                    possible_count++;
                    possible_option = option + 1; // Convert to 1-indexed
                }
            }
            
            // If only one option is possible, we can identify this voter
            if (possible_count == 1) {
                identified_voters.insert(voter_idx);
                whale_guesses[voter_idx] = possible_option;
                is_whale_guess[voter_idx] = true;
                total_whales++;
                total_whale_power += voter_weight;
                made_progress = true;
                
                // Check if guess is correct
                if (possible_option == actual_votes[idx]) {
                    correct_whales++;
                    correct_whale_power += voter_weight;
                }
                
                // Update tallies by removing this voter's contribution
                int actual_option_idx = actual_votes[idx] - 1; // Convert to 0-indexed
                if (actual_option_idx >= 0 && actual_option_idx < num_options) {
                    current_tallies[actual_option_idx] -= voter_weight;
                }
            }
        }
    }
    
    return {total_whales, correct_whales, total_whale_power, correct_whale_power};
}

// Standard meet-in-the-middle helper (for subset-sum mod MOD)
vector<uint64_t> computeSums(const vector<uint64_t>& arr, uint64_t MOD) {
    vector<uint64_t> sums = {0};
    for (uint64_t w : arr) {
        size_t sz = sums.size();
        for (size_t i = 0; i < sz; i++) {
            sums.push_back((sums[i] + w) % MOD);
        }
    }
    return sums;
}

bool meetInMiddleSubsetSum(const vector<uint64_t>& weights, uint64_t target, uint64_t MOD) {
    size_t n = weights.size();
    if (n == 0) return target == 0;
    
    size_t mid = n / 2;
    vector<uint64_t> left(weights.begin(), weights.begin() + mid);
    vector<uint64_t> right(weights.begin() + mid, weights.end());

    vector<uint64_t> leftSums = computeSums(left, MOD);
    vector<uint64_t> rightSums = computeSums(right, MOD);
    sort(rightSums.begin(), rightSums.end());

    for (uint64_t l : leftSums) {
        uint64_t rNeeded = (target + MOD - l) % MOD;
        if (binary_search(rightSums.begin(), rightSums.end(), rNeeded)) {
            return true;
        }
    }
    return false;
}

// Function to run the precision attack on remaining voters after whale attack
tuple<int, int, double, double> runPrecisionAttack(const Proposal& proposal, const vector<bool>& is_whale_guess) {
    // --- Simulation parameters ---
    uint64_t precision = 16;
    uint64_t modLowerBound = 9 * static_cast<uint64_t>(pow(10, precision));
    uint64_t modUpperBound = 10 * static_cast<uint64_t>(pow(10, precision));

    // Extract weights and votes for remaining voters (exclude whale guesses)
    vector<double> origWeights;
    vector<int> votes;
    vector<int> voter_indices; // To map back to original indices
    vector<vector<double>> option_weights(proposal.option_texts.size());

    const double scale = 1e18;
    for (size_t i = 0; i < proposal.voters.size(); i++) {
        const auto& voter = proposal.voters[i];
        if (voter.mapped_choice == 0) continue; // Skip unrecognized votes
        if (is_whale_guess[i]) continue; // Skip voters already identified by whale attack
        
        double flooredVotingPower = floor(voter.voting_power * scale);
        origWeights.push_back(flooredVotingPower);
        votes.push_back(voter.mapped_choice);
        voter_indices.push_back(i);
        
        // Add to appropriate option group
        if (voter.mapped_choice > 0 && voter.mapped_choice <= (int)option_weights.size()) {
            option_weights[voter.mapped_choice - 1].push_back(flooredVotingPower);
        }
    }

    if (origWeights.empty()) {
        return {0, 0, 0.0, 0.0}; // No remaining voters to analyze
    }

    // --- Setup random generator ---
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<uint64_t> modDist(modLowerBound, modUpperBound);

    // Lambda: compute the group tally mod current MOD
    auto computeGroupSumMod = [&](const vector<double>& groupWeights, uint64_t MOD) -> uint64_t {
        uint64_t sum = 0;
        for (double w : groupWeights) {
            uint64_t weightMod = static_cast<uint64_t>(floor(fmod(w, static_cast<double>(MOD))));
            sum = (sum + weightMod) % MOD;
        }
        return sum;
    };

    // --- Optimized simulation with iterative voter removal ---
    int totalVoters = origWeights.size();
    int guessedCount = 0;
    int correctGuesses = 0;
    double total_precision_power = 0.0;
    double correct_precision_power = 0.0;
    
    unordered_set<size_t> guessedVoters;
    vector<vector<double>> current_option_weights = option_weights;
    
    bool madeProgress = true;
    int iteration = 0;
    
    while (madeProgress && guessedVoters.size() < totalVoters) {
        madeProgress = false;
        iteration++;
        
        for (size_t i = 0; i < origWeights.size(); i++) {
            if (guessedVoters.count(i)) continue;
            
            vector<bool> deducible(proposal.option_texts.size(), false);
            
            // Test each option to see if voter i could have voted for it
            for (size_t optionIdx = 0; optionIdx < proposal.option_texts.size(); optionIdx++) {
                for (int attempt = 0; attempt < 3; attempt++) {
                    uint64_t MOD = modDist(gen);
                    uint64_t target = computeGroupSumMod(current_option_weights[optionIdx], MOD);
                    
                    // Build remaining weights excluding voter i and already guessed voters
                    vector<uint64_t> modWeights;
                    for (size_t j = 0; j < origWeights.size(); j++) {
                        if (guessedVoters.count(j)) continue;
                        if (j == i) continue;
                        
                        double modVal = fmod(origWeights[j], static_cast<double>(MOD));
                        uint64_t modInt = static_cast<uint64_t>(floor(modVal));
                        modWeights.push_back(modInt);
                    }
                    
                    // If target is NOT reachable without voter i, then voter i is essential
                    if (!meetInMiddleSubsetSum(modWeights, target, MOD)) {
                        deducible[optionIdx] = true;
                        break;
                    }
                }
            }
            
            // Count how many options are deducible
            int deducibleCount = 0;
            int deducibleOption = -1;
            for (size_t optionIdx = 0; optionIdx < deducible.size(); optionIdx++) {
                if (deducible[optionIdx]) {
                    deducibleCount++;
                    deducibleOption = optionIdx + 1; // 1-indexed
                }
            }
            
            // Make guess if exactly one option is deducible
            if (deducibleCount == 1) {
                guessedCount++;
                guessedVoters.insert(i);
                // Add voting power (convert back from scaled integer)
                double voter_power = origWeights[i] / scale;
                total_precision_power += voter_power;
                madeProgress = true;
                
                if (deducibleOption == votes[i]) {
                    correctGuesses++;
                    correct_precision_power += voter_power;
                }
                
                // Update current tallies by removing voter i's contribution
                int actualOption = votes[i] - 1; // Convert to 0-indexed
                if (actualOption >= 0 && actualOption < (int)current_option_weights.size()) {
                    auto& weights = current_option_weights[actualOption];
                    auto it = find(weights.begin(), weights.end(), origWeights[i]);
                    if (it != weights.end()) {
                        weights.erase(it);
                    }
                }
            }
        }
    }
    
    return {guessedCount, correctGuesses, total_precision_power, correct_precision_power};
}

// Function to run both whale and precision attacks on a proposal
AttackResult runCombinedAttack(const Proposal& proposal, int MAX_SIZE) {
    AttackResult result;
    result.proposal_id = proposal.proposal_id;
    result.dao_id = proposal.dao_id;
    result.total_voters = proposal.voters.size();
    result.option_texts = proposal.option_texts;
    result.option_counts = proposal.option_counts;
    result.option_votes = proposal.option_votes;
    
    // Count valid voters and total voting power
    result.valid_voters = 0;
    result.total_voting_power = 0.0;
    for (const auto& voter : proposal.voters) {
        if (voter.mapped_choice > 0) {
            result.valid_voters++;
            result.total_voting_power += voter.voting_power;
        }
    }
    
    // Initialize results
    result.whale_deanonymized = 0;
    result.whale_correct = 0;
    result.whale_voting_power = 0.0;
    result.whale_voting_power_correct = 0.0;
    result.precision_deanonymized = 0;
    result.precision_correct = 0;
    result.precision_voting_power = 0.0;
    result.precision_voting_power_correct = 0.0;
    result.total_deanonymized = 0;
    result.total_correct = 0;
    result.total_voting_power_deanonymized = 0.0;
    result.total_voting_power_correct = 0.0;
    result.accuracy = 0.0;
    
    // Count distinct options with voters
    int optionsWithVoters = 0;
    for (int count : proposal.option_counts) {
        if (count > 0) optionsWithVoters++;
    }
    
    // Only run attacks if there are options with voters
    if (optionsWithVoters >= 1 && result.valid_voters > 0) {
        // Step 1: Run whale attack (no size limit)
        vector<int> whale_guesses;
        vector<bool> is_whale_guess;
        auto [whale_count, whale_correct, whale_power, whale_power_correct] = runWhaleAttack(proposal, whale_guesses, is_whale_guess);
        result.whale_deanonymized = whale_count;
        result.whale_correct = whale_correct;
        result.whale_voting_power = whale_power;
        result.whale_voting_power_correct = whale_power_correct;
        
        // Step 2: Run precision attack only if remaining voters <= MAX_SIZE and noise flag is not set
        int remaining_voters = result.valid_voters - whale_count;
        if (remaining_voters <= MAX_SIZE && remaining_voters > 0 && !g_addNoise) {
            cout << "Precision" << endl;
            auto [precision_count, precision_correct, precision_power, precision_power_correct] = runPrecisionAttack(proposal, is_whale_guess);
            result.precision_deanonymized = precision_count;
            result.precision_correct = precision_correct;
            result.precision_voting_power = precision_power;
            result.precision_voting_power_correct = precision_power_correct;
        }
        
        // Calculate totals
        result.total_deanonymized = result.whale_deanonymized + result.precision_deanonymized;
        result.total_correct = result.whale_correct + result.precision_correct;
        result.total_voting_power_deanonymized = result.whale_voting_power + result.precision_voting_power;
        result.total_voting_power_correct = result.whale_voting_power_correct + result.precision_voting_power_correct;
        result.accuracy = (result.total_deanonymized > 0) ? 
                         (100.0 * result.total_correct / result.total_deanonymized) : 0.0;
    }
    
    return result;
}

// Function to escape CSV field
string escapeCSV(const string& field) {
    string cleaned = field;
    
    // Replace newlines and carriage returns with spaces
    for (size_t i = 0; i < cleaned.length(); i++) {
        if (cleaned[i] == '\n' || cleaned[i] == '\r') {
            cleaned[i] = ' ';
        }
    }
    
    // Remove any trailing/leading whitespace
    while (!cleaned.empty() && (cleaned.back() == ' ' || cleaned.back() == '\t')) {
        cleaned.pop_back();
    }
    while (!cleaned.empty() && (cleaned.front() == ' ' || cleaned.front() == '\t')) {
        cleaned.erase(0, 1);
    }
    
    if (cleaned.find(',') != string::npos || cleaned.find('"') != string::npos) {
        string escaped = "\"";
        for (char c : cleaned) {
            if (c == '"') escaped += "\"\"";
            else escaped += c;
        }
        escaped += "\"";
        return escaped;
    }
    return cleaned;
}

int main(int argc, char* argv[]) {
    const int MAX_SIZE = 45; // Maximum proposal size to run attack on
    
    // Parse command line arguments
    int iterations = 1;  // Default to 1 iteration
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--add-noise") == 0) {
            g_addNoise = true;
        } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            iterations = atoi(argv[i + 1]);
            i++; // Skip the next argument since we used it
        }
    }

    if (g_addNoise) {
        cout << "Running with noise added to tallies (precision attack disabled)" << endl;
        cout << "Running " << iterations << " iterations and averaging results" << endl;
    }
    
    // Initialize random number generator
    random_device rd;
    g_rng.seed(rd());
    
    // --- Read input and group by proposal_id ---
    ifstream file("../data_input/all_snapshot.csv");
    if (!file.is_open()) {
        cerr << "Error opening file!" << endl;
        return 1;
    }
    
    string line;
    getline(file, line); // skip header

    map<string, Proposal> proposals;

    while (getline(file, line)) {
        // Parse CSV line with proper quote handling
        vector<string> fields;
        string field;
        bool inQuotes = false;
        
        for (size_t i = 0; i < line.length(); i++) {
            char c = line[i];
            if (c == '"') {
                inQuotes = !inQuotes;
            } else if (c == ',' && !inQuotes) {
                if (!field.empty() && field.front() == '"' && field.back() == '"') {
                    field = field.substr(1, field.length() - 2);
                }
                fields.push_back(field);
                field.clear();
            } else {
                field += c;
            }
        }
        if (!field.empty() && field.front() == '"' && field.back() == '"') {
            field = field.substr(1, field.length() - 2);
        }
        fields.push_back(field);
        
        if (fields.size() < 6) continue;
        
        string platform = fields[0];
        string proposal_id = fields[1];
        string voter_address = fields[2];
        string choice = fields[3];
        string votingPowerStr = fields[4];
        string dao_id = fields[5];
        
        double votingPower = stod(votingPowerStr);
        
        // Create voter
        Voter voter;
        voter.voter_address = voter_address;
        voter.choice = choice;
        voter.voting_power = votingPower;
        voter.mapped_choice = 0; // Will be set later
        
        // Add to proposal
        if (proposals.find(proposal_id) == proposals.end()) {
            proposals[proposal_id] = Proposal{proposal_id, dao_id, {}, 0.0, {}, {}};
        }
        proposals[proposal_id].voters.push_back(voter);
        proposals[proposal_id].total_votes += votingPower;
    }
    file.close();

    // Map choices for each proposal
    for (auto& [proposal_id, proposal] : proposals) {
        mapChoicesForProposal(proposal);
    }
    
    cout << "Processing " << proposals.size() << " proposals..." << endl;
    
    // Process each proposal and store results
    vector<AttackResult> results;
    
    if (g_addNoise && iterations > 1) {
        // For noise experiments with multiple iterations, average the results
        map<string, vector<AttackResult>> iteration_results;
        
        for (int iter = 0; iter < iterations; iter++) {
            cout << "Running iteration " << (iter + 1) << "/" << iterations << "..." << endl;
            int processedCount = 0;
            
            for (const auto& [proposal_id, proposal] : proposals) {
                processedCount++;
                if (processedCount % 100 == 0) {
                    cout << "  Processed " << processedCount << " / " << proposals.size() << " proposals..." << endl;
                }
                
                AttackResult result = runCombinedAttack(proposal, MAX_SIZE);
                iteration_results[proposal_id].push_back(result);
            }
        }
        
        // Average the results across iterations
        cout << "Averaging results across " << iterations << " iterations..." << endl;
        for (const auto& [proposal_id, proposal] : proposals) {
            const auto& iter_results = iteration_results[proposal_id];
            AttackResult averaged_result = iter_results[0]; // Start with first iteration
            
            // Average all numeric fields
            averaged_result.whale_deanonymized = 0;
            averaged_result.whale_correct = 0;
            averaged_result.whale_voting_power = 0.0;
            averaged_result.whale_voting_power_correct = 0.0;
            averaged_result.precision_deanonymized = 0;
            averaged_result.precision_correct = 0;
            averaged_result.precision_voting_power = 0.0;
            averaged_result.precision_voting_power_correct = 0.0;
            averaged_result.total_deanonymized = 0;
            averaged_result.total_correct = 0;
            averaged_result.total_voting_power_deanonymized = 0.0;
            averaged_result.total_voting_power_correct = 0.0;
            averaged_result.accuracy = 0.0;
            
            for (const auto& result : iter_results) {
                averaged_result.whale_deanonymized += result.whale_deanonymized;
                averaged_result.whale_correct += result.whale_correct;
                averaged_result.whale_voting_power += result.whale_voting_power;
                averaged_result.whale_voting_power_correct += result.whale_voting_power_correct;
                averaged_result.precision_deanonymized += result.precision_deanonymized;
                averaged_result.precision_correct += result.precision_correct;
                averaged_result.precision_voting_power += result.precision_voting_power;
                averaged_result.precision_voting_power_correct += result.precision_voting_power_correct;
                averaged_result.total_deanonymized += result.total_deanonymized;
                averaged_result.total_correct += result.total_correct;
                averaged_result.total_voting_power_deanonymized += result.total_voting_power_deanonymized;
                averaged_result.total_voting_power_correct += result.total_voting_power_correct;
                averaged_result.accuracy += result.accuracy;
            }
            
            // Divide by number of iterations to get average
            int num_iterations = iter_results.size();
            averaged_result.whale_deanonymized = (int)round((double)averaged_result.whale_deanonymized / num_iterations);
            averaged_result.whale_correct = (int)round((double)averaged_result.whale_correct / num_iterations);
            averaged_result.whale_voting_power /= num_iterations;
            averaged_result.whale_voting_power_correct /= num_iterations;
            averaged_result.precision_deanonymized = (int)round((double)averaged_result.precision_deanonymized / num_iterations);
            averaged_result.precision_correct = (int)round((double)averaged_result.precision_correct / num_iterations);
            averaged_result.precision_voting_power /= num_iterations;
            averaged_result.precision_voting_power_correct /= num_iterations;
            averaged_result.total_deanonymized = (int)round((double)averaged_result.total_deanonymized / num_iterations);
            averaged_result.total_correct = (int)round((double)averaged_result.total_correct / num_iterations);
            averaged_result.total_voting_power_deanonymized /= num_iterations;
            averaged_result.total_voting_power_correct /= num_iterations;
            averaged_result.accuracy /= num_iterations;
            
            results.push_back(averaged_result);
        }
    } else {
        // Single iteration (normal case or no noise)
        int processedCount = 0;
        
        for (const auto& [proposal_id, proposal] : proposals) {
            processedCount++;
            if (processedCount % 100 == 0) {
                cout << "Processed " << processedCount << " / " << proposals.size() << " proposals..." << endl;
            }
            
            AttackResult result = runCombinedAttack(proposal, MAX_SIZE);
            results.push_back(result);
        }
    }
    
    // Save results to CSV
    string filename = g_addNoise ? "privacy_attack_results_noise.csv" : "privacy_attack_results.csv";
    ofstream outFile(filename);
    if (!outFile.is_open()) {
        cerr << "Error creating output file!" << endl;
        return 1;
    }
    
    // Write header
    outFile << "proposal_id,dao_id,total_voters,valid_voters,total_voting_power,num_options,";
    outFile << "whale_deanonymized,whale_correct,whale_voting_power,whale_voting_power_correct,";
    outFile << "precision_deanonymized,precision_correct,precision_voting_power,precision_voting_power_correct,";
    outFile << "total_deanonymized,total_correct,total_voting_power_deanonymized,total_voting_power_correct,accuracy,";
    outFile << "option1_text,option1_count,option1_votes,option2_text,option2_count,option2_votes,option3_text,option3_count,option3_votes,option4_text,option4_count,option4_votes,option5_text,option5_count,option5_votes" << endl;
    
    // Write results
    for (const auto& result : results) {
        outFile << escapeCSV(result.proposal_id) << ",";
        outFile << escapeCSV(result.dao_id) << ",";
        outFile << result.total_voters << ",";
        outFile << result.valid_voters << ",";
        outFile << fixed << setprecision(6) << result.total_voting_power << ",";
        outFile << result.option_texts.size() << ",";
        outFile << result.whale_deanonymized << ",";
        outFile << result.whale_correct << ",";
        outFile << fixed << setprecision(6) << result.whale_voting_power << ",";
        outFile << fixed << setprecision(6) << result.whale_voting_power_correct << ",";
        outFile << result.precision_deanonymized << ",";
        outFile << result.precision_correct << ",";
        outFile << fixed << setprecision(6) << result.precision_voting_power << ",";
        outFile << fixed << setprecision(6) << result.precision_voting_power_correct << ",";
        outFile << result.total_deanonymized << ",";
        outFile << result.total_correct << ",";
        outFile << fixed << setprecision(6) << result.total_voting_power_deanonymized << ",";
        outFile << fixed << setprecision(6) << result.total_voting_power_correct << ",";
        outFile << fixed << setprecision(2) << result.accuracy << ",";
        
        // Write up to 5 options (can be extended if needed)
        for (int i = 0; i < 5; i++) {
            if (i < (int)result.option_texts.size()) {
                outFile << escapeCSV(result.option_texts[i]) << ",";
                outFile << result.option_counts[i] << ",";
                outFile << fixed << setprecision(2) << result.option_votes[i];

            } else {
                outFile << ",0";
            }
            if (i < 4) outFile << ",";
        }
        outFile << endl;
    }
    
    outFile.close();
    
    cout << "Results saved to " << filename << endl;
    cout << "Total proposals: " << results.size() << endl;
    
    // Print summary statistics
    int attackedProposals = 0;
    int whaleSuccesses = 0;
    int precisionSuccesses = 0;
    int totalDeanonymized = 0;
    int totalWhales = 0;
    int totalPrecision = 0;
    double totalVotingPowerDeanonymized = 0.0;
    double totalWhaleVotingPower = 0.0;
    double totalPrecisionVotingPower = 0.0;
    double totalValidVotingPower = 0.0;
    
    for (const auto& result : results) {
        totalValidVotingPower += result.total_voting_power;
        
        if (result.total_deanonymized > 0) {
            attackedProposals++;
            totalDeanonymized += result.total_deanonymized;
            totalVotingPowerDeanonymized += result.total_voting_power_deanonymized;
        }
        if (result.whale_deanonymized > 0) {
            whaleSuccesses++;
            totalWhales += result.whale_deanonymized;
            totalWhaleVotingPower += result.whale_voting_power;
        }
        if (result.precision_deanonymized > 0) {
            precisionSuccesses++;
            totalPrecision += result.precision_deanonymized;
            totalPrecisionVotingPower += result.precision_voting_power;
        }
    }
    
    cout << "Proposals with successful attacks: " << attackedProposals << endl;
    cout << "Proposals with whale attacks: " << whaleSuccesses << endl;
    cout << "Proposals with precision attacks: " << precisionSuccesses << " (only run on ≤" << MAX_SIZE << " remaining voters)" << endl;
    cout << "Total voters deanonymized: " << totalDeanonymized << endl;
    cout << "  - By whale attacks: " << totalWhales << endl;
    cout << "  - By precision attacks: " << totalPrecision << endl;
    cout << "Total voting power deanonymized: " << fixed << setprecision(2) << totalVotingPowerDeanonymized << endl;
    cout << "  - By whale attacks: " << fixed << setprecision(2) << totalWhaleVotingPower << endl;
    cout << "  - By precision attacks: " << fixed << setprecision(2) << totalPrecisionVotingPower << endl;
    cout << "Total valid voting power: " << fixed << setprecision(2) << totalValidVotingPower << endl;
    cout << "Percentage of voting power deanonymized: " << fixed << setprecision(2) << 
            (100.0 * totalVotingPowerDeanonymized / totalValidVotingPower) << "%" << endl;
    
    return 0;
}