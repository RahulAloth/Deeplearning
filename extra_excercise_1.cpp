#include <algorithm>
#include <numeric>
#include <vector>
#include <iterator>
#include <iostream>
#include <random>
#include <ranges>

// Select elements and copy them to a new vector.
//
// This version of "select" can only run sequentially, because the output
// vector w is built consecutively during the traversal of the input vector v.
template<class UnaryPredicate>
std::vector<int> select(const std::vector<int>& v, UnaryPredicate pred)
{
    std::vector<int> w;
    // Reserve capacity to avoid multiple reallocations
    w.reserve(v.size());

    // Copy elements from v to w that satisfy the predicate
    std::copy_if(v.begin(), v.end(), std::back_inserter(w), pred);

    return w;
}

// Initialize vector
void initialize(std::vector<int>& v);

int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::cerr << "ERROR: Missing length argument!" << std::endl;
        return 1;
    }

    long long n = std::stoll(argv[1]);

    auto v = std::vector<int>(n);

    initialize(v);

    auto predicate = [](int x) { return x % 3 == 0; };
    auto w = select(v, predicate);

    if (!std::all_of(w.begin(), w.end(), predicate) || w.empty()) {
        std::cerr << "ERROR!" << std::endl;
        return 1;
    }
    std::cerr << "OK!" << std::endl;

    std::cout << "w = ";
    std::copy(w.begin(), w.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;

    return 0;
}

void initialize(std::vector<int>& v)
{
    auto distribution = std::uniform_int_distribution<int> {0, 100};
    auto engine = std::mt19937 {1};
    std::generate(v.begin(), v.end(), [&distribution, &engine]{ return distribution(engine); });
}
