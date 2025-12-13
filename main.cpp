#include <iostream>
#include <vector>
#include <cassert>
#include <algorithm>

using namespace std;

int main()
{
    cout << "C++  ISO C++ Algorithms" << endl;
    cout << "=======================================" << endl;

    std::vector<double> v = {1, 2, 3, 4};
    double s = 2;

    // Create a result vector to store transformed values
    std::vector<double> result(v.size());

    // Use std::transform with lambda to multiply each element by s
    std::transform(v.begin(), v.end(), result.begin(),
                   [s](double val) { return val * s; });

    // Verify the transformation worked correctly
    assert(result[1] == 4);

    cout << "Original vector: ";
    for (auto val : v) {
        cout << val << " ";
    }
    cout << endl;

    cout << "Vector elements multiplied by " << s << " using std::transform:" << endl;
    for (size_t i = 0; i < result.size(); ++i)
    {
        cout << "result[" << i << "] = " << result[i] << endl;
    }

    cout << "Assertion passed: result[1] == " << result[1] << endl;
    cout << "\nDemonstration completed!" << endl;
    return 0;
}