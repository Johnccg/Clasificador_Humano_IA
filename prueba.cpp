#include <iostream>
#include <string>

using namespace std;

int main() {
    string arr[] = {""apple"", ""orange""};
    cout << sizeof(arr) / sizeof(arr[0]) << endl;
    return 0;
}",0
0,"#include <iostream>
#include <string>
#include <boost/multiprecision/cpp_int.hpp>

using namespace boost::multiprecision;

int main() {
    cpp_int power = pow(cpp_int(5), pow(cpp_int(4), pow(cpp_int(3), 2).convert_to<int>()).convert_to<int>());
    std::string str = power.str();
    int len = str.length();
    std::cout << ""5**4**3**2 = "" << str.substr(0, 20) << ""..."" << str.substr(len - 20) << "" and has "" << len << "" digits"" << std::endl;
}